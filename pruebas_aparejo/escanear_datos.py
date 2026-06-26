"""
Escanea todas las páginas de la API para contar cuántas imágenes
hay en total para 'Conforme' y 'Dado de baja'.
"""
from collections import defaultdict
from kull_api import KullClient

def main():
    c = KullClient()
    counts = defaultdict(int)
    img_counts = defaultdict(int)
    page = 1
    
    print("Escaneando API de Kull para contar productos e imágenes...")
    while True:
        try:
            res = c.productos(limit=200, page=page)
            items = res["data"]["items"]
        except Exception as e:
            print(f"Error en página {page}: {e}")
            break
            
        if not items:
            print(f"Fin de los datos en la página {page}.")
            break
            
        for p in items:
            estado = p.get("statusLabel")
            if not estado:
                continue
            counts[estado] += 1
            img_counts[estado] += len(p.get("imagenes", []))
            
        print(f"Página {page} procesada. Conforme (imgs): {img_counts['Conforme']} | Dado de baja (imgs): {img_counts['Dado de baja']}")
        page += 1

    print("\n=== RECUENTO TOTAL ===")
    for estado in set(counts.keys()).union(img_counts.keys()):
        print(f"Estado: {estado} | Productos: {counts[estado]} | Imágenes: {img_counts[estado]}")

if __name__ == "__main__":
    main()
