pipeline {
    agent any
    // Definir workspaceDir fuera de los bloques stages
    environment {
        workspaceDir = pwd()
    }
    parameters {
        string(name: 'URI', defaultValue: '', description: 'URI a utilizar')
        string(name: 'nombre_experimento', defaultValue: '', description: 'Nombre del experimento')
    }
    stages {
        stage('Acceso a Dockerfile en el Workspace') {
            steps {
                script {
                    // Definir la ruta completa al Dockerfile en el workspace
                    def rutaDockerfile = "${workspaceDir}/Dockerfile"

                    // Verificar si el Dockerfile existe en el workspace
                    if (fileExists(rutaDockerfile)) {
                        echo "El Dockerfile existe en la ruta: ${rutaDockerfile}"
                        
                        // Realizar acciones adicionales con el Dockerfile si es necesario
                    } else {
                        echo "El Dockerfile no se encontró en la ruta: ${rutaDockerfile}"
                    }
                }
            }
        }
        stage('Eliminar imagenes y contenedores previos en Docker') {
            steps {
                script {
                    // Detener y eliminar todos los contenedores
                    sh "docker stop \$(docker ps -a -q) || true"
                    sh "docker rm \$(docker ps -a -q) || true"

                    // Eliminar todas las imágenes
                    sh "docker rmi \$(docker images -q) || true"
                }
            }
        }

        stage('Docker Compose') {
            steps {
                script {
                    // Ejecuta el docker-compose
                    sh "URI=${params.URI} nombre_experimento=${params.nombre_experimento} docker-compose -f docker-compose.yaml up -d"
                    }
            }
        }

    }
}
