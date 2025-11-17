# 1. Detener los contenedores actuales
docker-compose down

# 2. construir las imágenes con las  dependencias
docker-compose build --no-cache

# 3. Iniciar de nuevo
docker-compose up -d

# 4. Ver los logs para confirmar que ya no hay errores
docker-compose logs -f

# verificar que funciona
curl http://localhost/health

# Ver todos los contenedores
docker-compose ps

# Ver logs en tiempo real
docker-compose logs -f

# Ver logs de un nodo específico
docker-compose logs -f api_node1
docker-compose logs -f nginx

# Ver qué nodo responde cada vez
curl -I http://localhost/health | grep X-Upstream-Server

# Hacer múltiples requests y ver la distribución
for i in {1..20}; do
  curl -s -I http://localhost/health | grep X-Upstream-Server
done

# Detener un nodo para ver el auto-recovery
docker-compose stop api_node1

# Nginx automáticamente redirige a los otros 5 nodos

# Volver a iniciar el nodo
docker-compose start api_node1
