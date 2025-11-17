# -*- coding: utf-8 -*-
"""Test de Load Balancing - Version Simple para Windows"""

import requests
import time
import concurrent.futures
from collections import Counter

BASE_URL = "http://localhost"


def make_request():
    """Realiza una solicitud al endpoint de health"""
    try:
        response = requests.get(f"{BASE_URL}/health", timeout=5)
        return {
            'status': 'success',
            'upstream': response.headers.get('X-Upstream-Server', 'unknown'),
            'status_code': response.status_code,
            'response_time': response.elapsed.total_seconds()
        }
    except Exception as e:
        return {'status': 'error', 'upstream': 'error', 'error': str(e)}


def test_concurrent(num_requests, workers):
    """Prueba concurrente"""
    print(f"\n[*] Ejecutando {num_requests} solicitudes con {workers} workers...\n")

    start_time = time.time()
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
        results = list(executor.map(lambda _: make_request(), range(num_requests)))

    elapsed = time.time() - start_time
    print(f"[OK] Completado en {elapsed:.2f} segundos")
    print(f"[*] Throughput: {num_requests / elapsed:.2f} req/s\n")

    return results


def analyze_results(results):
    """Analiza y muestra estadísticas"""
    successes = [r for r in results if r['status'] == 'success']
    errors = [r for r in results if r['status'] == 'error']

    print("=" * 60)
    print("RESULTADOS DEL TEST")
    print("=" * 60)
    print(f"\n[OK] Exitosas: {len(successes)}/{len(results)}")
    print(f"[ERROR] Fallidas: {len(errors)}/{len(results)}")

    if not successes:
        return

    # Distribución por nodo
    print("\n" + "=" * 60)
    print("DISTRIBUCION DE CARGA POR NODO")
    print("=" * 60 + "\n")

    node_distribution = Counter([r['upstream'] for r in successes])
    total = len(successes)

    print(f"{'Nodo':<35} {'Requests':<10} {'Porcentaje'}")
    print("-" * 60)

    for node, count in sorted(node_distribution.items(), key=lambda x: x[1], reverse=True):
        percentage = (count / total) * 100
        bar = "#" * int(percentage / 2)
        print(f"{node:<35} {count:<10} {percentage:>6.2f}%  {bar}")

    # Estadísticas
    counts = list(node_distribution.values())
    if len(counts) > 1:
        avg = sum(counts) / len(counts)
        std_dev = (sum((x - avg) ** 2 for x in counts) / len(counts)) ** 0.5
        cv = (std_dev / avg) * 100

        print("\n" + "=" * 60)
        print("ESTADISTICAS")
        print("=" * 60)
        print(f"\nPromedio por nodo: {avg:.2f}")
        print(f"Desviacion estandar: {std_dev:.2f}")
        print(f"Coeficiente de variacion: {cv:.2f}%")

        if cv < 15:
            print("\n[OK] Excelente distribucion (CV < 15%)")
        elif cv < 30:
            print("\n[!] Distribucion aceptable (CV < 30%)")
        else:
            print("\n[!] Distribucion desigual (CV > 30%)")

    # Tiempos de respuesta
    response_times = [r['response_time'] for r in successes if 'response_time' in r]
    if response_times:
        print("\n" + "=" * 60)
        print("TIEMPOS DE RESPUESTA")
        print("=" * 60)

        avg_time = sum(response_times) / len(response_times)
        min_time = min(response_times)
        max_time = max(response_times)

        sorted_times = sorted(response_times)
        p50 = sorted_times[len(sorted_times) // 2]
        p95 = sorted_times[int(len(sorted_times) * 0.95)]

        print(f"\nPromedio: {avg_time * 1000:.2f} ms")
        print(f"Minimo:   {min_time * 1000:.2f} ms")
        print(f"Maximo:   {max_time * 1000:.2f} ms")
        print(f"P50:      {p50 * 1000:.2f} ms")
        print(f"P95:      {p95 * 1000:.2f} ms")


def main():
    print("=" * 60)
    print("CAT VS DOG - TEST DE LOAD BALANCING")
    print("=" * 60)

    # Verificar sistema
    print("\n[*] Verificando sistema...")
    try:
        response = requests.get(f"{BASE_URL}/health", timeout=5)
        if response.status_code != 200:
            print("[ERROR] Sistema no disponible")
            return
        print("[OK] Sistema operativo\n")
    except Exception as e:
        print(f"[ERROR] No se pudo conectar: {e}")
        return

    # Menu
    print("=" * 60)
    print("OPCIONES")
    print("=" * 60)
    print("\n1. Test rapido (20 requests)")
    print("2. Test normal (100 requests)")
    print("3. Test intensivo (500 requests)")
    print("0. Salir")

    try:
        choice = input("\nSelecciona [1-3]: ").strip()

        if choice == "1":
            results = test_concurrent(20, 5)
        elif choice == "2":
            results = test_concurrent(100, 10)
        elif choice == "3":
            results = test_concurrent(500, 20)
        elif choice == "0":
            print("\nSaliendo...")
            return
        else:
            print("\n[ERROR] Opcion invalida")
            return

        analyze_results(results)

        print("\n" + "=" * 60)
        print("TEST COMPLETADO")
        print("=" * 60)
        print("\n[*] Dashboard: http://localhost:8081")
        print("[*] Nginx stats: http://localhost:8081/nginx_status\n")

    except KeyboardInterrupt:
        print("\n\n[!] Test interrumpido")
    except Exception as e:
        print(f"\n[ERROR] {e}")


if __name__ == "__main__":
    main()