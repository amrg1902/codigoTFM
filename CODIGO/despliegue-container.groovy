pipeline {
    agent any // Utiliza cualquier agente disponible en Jenkins

    stages {
        stage('Crear Contenedor NGINX') {
            steps {
                script {
                    def nginxImage = 'nginx:latest' // Puedes cambiar la etiqueta de la imagen si es necesario
                    // Iniciar un temporizador de 30 segundos (puedes ajustar el tiempo según tus necesidades)
                    def timeoutSeconds = 30
                    def startTime = currentBuild.startTimeInMillis

                    // Ejecuta el contenedor NGINX con Docker
                    sh "docker run -d -p 80:80 --name mi-contenedor-nginx $nginxImage"
                    
                    // Comprueba si ha pasado el tiempo especificado
                    def elapsedTime = (currentBuild.startTimeInMillis - startTime) / 1000
                    if (elapsedTime >= timeoutSeconds) {
                        error "Tiempo de ejecución excedido. Cerrando la etapa."
                    
                    }
                }
            }
        }
    }
}
