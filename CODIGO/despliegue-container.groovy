pipeline {
    agent any // Utiliza cualquier agente disponible en Jenkins

    stages {
        stage('Crear Contenedor NGINX') {
            steps {
                script {
                    def nginxImage = 'nginx:latest' // Puedes cambiar la etiqueta de la imagen si es necesario

                    // Ejecuta el contenedor NGINX con Docker
                    sh "docker run -d -p 80:80 --name mi-contenedor-nginx $nginxImage"
                }
            }
        }
    }
}
