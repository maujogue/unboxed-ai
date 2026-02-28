import requests

ORTHANC = "https://orthanc.unboxed-2026.ovh"
AUTH    = ("unboxed", "unboxed2026")

print("─" * 50)
try:
    r = requests.get(f"{ORTHANC}/system", auth=AUTH, timeout=5)
    if r.status_code == 200:
        info = r.json()
        print(f"  ✅ Orthanc connecté")
        print(f"  Version     : {info['Version']}")
        print(f"  AET         : {info['DicomAet']}")
        print(f"  Storage     : {info.get('StorageSize', 'N/A')}")
    else:
        print(f"  ❌ Erreur HTTP {r.status_code}")
except requests.exceptions.ConnectionError:
    print("  ❌ Impossible de joindre Orthanc (réseau privé requis)")
except requests.exceptions.Timeout:
    print("  ❌ Timeout — serveur Orthanc non joignable")
print("─" * 50)