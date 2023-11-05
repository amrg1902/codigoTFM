pipeline {
    agent any
    // Definir workspaceDir fuera de los bloques stages
    environment {
        workspaceDir = pwd()
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
        stage('Construir imagen Docker training') {
            steps {
                script {
                    def dockerfilePath = "${workspaceDir}/training-container/Dockerfile"
                    def dockerImageName = "mi_imagen_docker:latest"
                    sh "docker build -t ${dockerImageName} -f ${dockerfilePath} ."
                }
            }
        }
        stage('Construir imagen Docker mlflow') {
            steps {
                script {
                    def dockerfilePath = "${workspaceDir}/mlflow-container/Dockerfile"
                    def dockerImageName = "mi_imagen_docker:latest"
                    sh "docker build -t ${dockerImageName} -f ${dockerfilePath} ."
                }
            }
        }
        // stage('Desplegar contenedor') {
        //     steps {
        //         script {
        //             def dockerImageName = "mi_imagen_docker:latest"
        //             def containerName = "mi_contenedor"

        //             // Detener y eliminar el contenedor si ya existe (opcional)
        //             sh "docker stop ${containerName} || true"
        //             sh "docker rm ${containerName} || true"

        //             // Ejecutar el contenedor a partir de la imagen
        //             sh "docker run -d -p 5005:5005 --name ${containerName} ${dockerImageName}"
        //         }
        //     }
        // }


    }
}
