"""
Cliente para la API de Doland/Kull.
Inicia sesión automáticamente y mantiene un token vigente
(el accessToken expira cada 3600 s).
"""

import requests

BASE = "https://api-doland.kull.cl/api"
EMAIL = "soporte@kull.cl"
PASSWORD = "123456"


class KullClient:
    def __init__(self, email=EMAIL, password=PASSWORD):
        self.email = email
        self.password = password
        self._token = None
        self.s = requests.Session()

    def login(self):
        r = self.s.post(f"{BASE}/auth/login", json={"email": self.email, "password": self.password}, timeout=30)
        r.raise_for_status()
        self._token = r.json()["data"]["accessToken"]
        return self._token

    def _headers(self):
        if not self._token:
            self.login()
        return {"Authorization": f"Bearer {self._token}"}

    def get(self, path, params=None):
        """GET con reintento automático si el token expiró (401)."""
        url = f"{BASE}/{path.lstrip('/')}"
        r = self.s.get(url, headers=self._headers(), params=params, timeout=60)
        if r.status_code == 401:
            self.login()
            r = self.s.get(url, headers=self._headers(), params=params, timeout=60)
        r.raise_for_status()
        return r.json()

    def productos(self, cliente_id=703, anio=2026, limit=10, page=1):
        return self.get(
            "informes/productos", params={"clienteId": cliente_id, "anio": anio, "limit": limit, "page": page}
        )


if __name__ == "__main__":
    import json

    c = KullClient()
    data = c.productos()
    print(json.dumps(data, indent=2, ensure_ascii=False)[:3000])
