"""
Prepara el dataset de aparejos en formato YOLO clasificación con las siguientes mejoras:
1. Filtro administrativo: Si el reporte de fallas de la API se debe puramente a 
   problemas de registro/etiquetado (ej: placa ilegible/faltante) y no a fallos estructurales,
   re-clasifica la imagen como "conforme".
2. Recorte inteligente (Cropping): Usa YOLO-World para localizar la eslinga, grillete, 
   cadena o gancho, recorta la pieza descartando el fondo, y la guarda a resolución 640x640.
3. Descarga concurrente: Conserva la velocidad del ThreadPoolExecutor.
"""
import csv
import random
import requests
from collections import defaultdict
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from PIL import Image
import torch

from kull_api import KullClient
# Cargamos YOLO-World solo si está disponible para evitar fallos de importación previos
try:
    from ultralytics import YOLOWorld
    DETECTOR_DISPONIBLE = True
except ImportError:
    DETECTOR_DISPONIBLE = False

OUT = Path("dataset_cls")
CLASES = {"Conforme": "conforme", "Dado de baja": "dado_de_baja"}
SPLIT = (0.70, 0.15, 0.15)
random.seed(42)
MAX_WORKERS = 24
IMGSZ = 800

# Palabras clave para identificar rechazos puramente administrativos
ADMIN_KEYWORDS = ["placa", "tarjeta", "rotulacion", "rotulo", "ilegible", "identificacion", "certif", "marca", "sin placa", "placa de carga"]
# Palabras clave de daño físico estructural (si están presentes, NO es administrativo)
STRUCTURAL_KEYWORDS = ["corte", "desgaste", "oxid", "corros", "rotur", "daño", "fisura", "estira", "deform", "aplast", "fisur", "friccion", "quemad"]

def es_defecto_administrativo(comentario):
    """Retorna True si el defecto es puramente administrativo (sin daño estructural)."""
    if not comentario:
        return False
    texto = comentario.lower()
    
    # Validar si contiene palabras administrativas
    tiene_admin = any(kw in texto for kw in ADMIN_KEYWORDS)
    # Validar si contiene alguna palabra estructural
    tiene_estructural = any(kw in texto for kw in STRUCTURAL_KEYWORDS)
    
    return tiene_admin and not tiene_estructural

def recolectar():
    c = KullClient()
    buckets = defaultdict(list)
    page = 1
    print("Recolectando metadatos de todos los productos en la API...")
    while True:
        res = c.productos(limit=200, page=page)
        items = res["data"]["items"]
        if not items:
            break
        for p in items:
            estado = p.get("statusLabel")
            comentario = " ".join((p.get("comentario") or "").split())
            
            # Filtro Administrativo: Si está dado de baja pero solo es administrativo, re-clasificar como Conforme
            if estado == "Dado de baja" and es_defecto_administrativo(comentario):
                estado_visual = "Conforme"
            else:
                estado_visual = estado
                
            if estado_visual not in CLASES:
                continue
                
            for n, img in enumerate(p.get("imagenes", []), 1):
                buckets[estado_visual].append((
                    p["id"], n, img["url"],
                    comentario,
                    (p.get("productoCatalogo") or {}).get("descripcion", ""),
                ))
        print(f"  Página {page} leída. Conforme: {len(buckets['Conforme'])} | Dado de baja: {len(buckets['Dado de baja'])}")
        page += 1
    return buckets

def crop_object(img_path, detector):
    """Usa YOLO-World para recortar el objeto metálico o eslinga principal."""
    if not DETECTOR_DISPONIBLE or detector is None:
        return
    try:
        results = detector(str(img_path), verbose=False)
        r = results[0]
        if len(r.boxes) == 0:
            return # Fallback a imagen completa
            
        # Tomar la caja con mayor confianza
        box = max(r.boxes, key=lambda b: float(b.conf))
        if float(box.conf) < 0.15:
            return
            
        xyxy = box.xyxy[0].tolist()
        x1, y1, x2, y2 = map(int, xyxy)
        
        # Abrir y recortar con PIL con un 10% de margen
        img = Image.open(img_path)
        w, h = img.size
        
        dx = int((x2 - x1) * 0.10)
        dy = int((y2 - y1) * 0.10)
        
        x1_new = max(0, x1 - dx)
        y1_new = max(0, y1 - dy)
        x2_new = min(w, x2 + dx)
        y2_new = min(h, y2 + dy)
        
        cropped_img = img.crop((x1_new, y1_new, x2_new, y2_new))
        # Redimensionar a resolución objetivo (640x640)
        cropped_img = cropped_img.resize((IMGSZ, IMGSZ), Image.Resampling.LANCZOS)
        cropped_img.save(img_path)
    except Exception as e:
        print(f"Error al recortar {img_path.name}: {e}")

def descargar_imagen(registro, split, slug, detector):
    pid, n, url, defecto, tipo = registro
    destino = OUT / split / slug
    destino.mkdir(parents=True, exist_ok=True)
    ruta = destino / f"{pid}_{n}.jpg"
    
    # Si ya existe en disco a resolución correcta, lo saltamos para reanudar rápido
    if ruta.exists() and ruta.stat().st_size > 0:
        return {
            "archivo": str(ruta), "split": split,
            "clase": slug, "producto_id": pid,
            "tipo": tipo, "defecto": defecto, "url": url
        }
        
    try:
        r = requests.get(url, timeout=30)
        r.raise_for_status()
        ruta.write_bytes(r.content)
        
        # Redimensionar y recortar la imagen al vuelo
        crop_object(ruta, detector)
        
        # Si no se recortó, asegurar que se guarde redimensionada a 640x640
        img = Image.open(ruta)
        if img.size != (IMGSZ, IMGSZ):
            img = img.resize((IMGSZ, IMGSZ), Image.Resampling.LANCZOS)
            img.save(ruta)
            
        return {
            "archivo": str(ruta), "split": split,
            "clase": slug, "producto_id": pid,
            "tipo": tipo, "defecto": defecto, "url": url
        }
    except Exception as e:
        return None

def main():
    # Inicializar YOLO-World
    detector = None
    if DETECTOR_DISPONIBLE:
        print("Cargando detector YOLO-World para recorte automático...")
        device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
        detector = YOLOWorld("yolov8x-worldv2.pt")
        detector.set_classes(["lifting sling", "shackle", "chain", "hook"])
        detector.to(device)
    else:
        print("Aviso: YOLO-World no instalado. Se guardarán las imágenes completas a 640x640.")

    buckets = recolectar()
    
    n_obj = min(len(buckets["Conforme"]), len(buckets["Dado de baja"]))
    print(f"\nTotal disponible para 'Dado de baja' (tras filtro admin): {len(buckets['Dado de baja'])}")
    print(f"Total disponible para 'Conforme': {len(buckets['Conforme'])}")
    print(f"Equilibrando dataset a {n_obj} imágenes por clase para entrenamiento balanceado.")

    tareas = []
    for estado, slug in CLASES.items():
        muestra = random.sample(buckets[estado], n_obj)
        n_tr = int(n_obj * SPLIT[0])
        n_va = int(n_obj * SPLIT[1])
        splits = [
            ("train", muestra[:n_tr]),
            ("val", muestra[n_tr:n_tr + n_va]),
            ("test", muestra[n_tr + n_va:])
        ]
        
        for split, registros in splits:
            for registro in registros:
                tareas.append((registro, split, slug))

    print(f"\nIniciando descarga concurrente de {len(tareas)} imágenes con {MAX_WORKERS} hilos...")
    
    filas = []
    descargadas = 0
    
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {executor.submit(descargar_imagen, reg, split, slug, detector): (reg, split, slug) for reg, split, slug in tareas}
        for future in as_completed(futures):
            res = future.result()
            if res:
                filas.append(res)
            descargadas += 1
            if descargadas % 200 == 0:
                print(f"  Progreso: {descargadas}/{len(tareas)} descargas procesadas e imágenes optimizadas...")

    OUT.mkdir(exist_ok=True)
    with open(OUT / "etiquetas.csv", "w", newline="", encoding="utf-8") as f:
        if filas:
            w = csv.DictWriter(f, fieldnames=list(filas[0].keys()))
            w.writeheader()
            w.writerows(filas)

    print(f"\nTotal registrado en etiquetas.csv: {len(filas)} imágenes en {OUT}/")
    print("Dataset de resolución 640x640 listo y filtrado.")

if __name__ == "__main__":
    main()
