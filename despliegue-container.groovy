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
        stage('Construir contenedor Docker postgreSQL') {
            steps {
                script {
                    def dockerfilePath = "${workspaceDir}/mlflow-db/Dockerfile"
                    def dockerImageName = "postgresql:latest"
                    def dockerContainerName = "mlflow_postgres"

                    // Detener y eliminar el contenedor si ya existe
                    sh "docker stop ${dockerContainerName} || true"
                    sh "docker rm ${dockerContainerName} || true"

                    // Crear la imagen
                    sh "docker build -t ${dockerImageName} -f ${dockerfilePath} ."

                    // Levantar el nuevo contenedor
                    sh "docker run --name ${dockerContainerName} -d ${dockerImageName}"
                }
            }
        }

        stage('Construir contenedor Docker mlflow') {
            steps {
                script {
                    def dockerfilePath = "${workspaceDir}/mlflow-container/Dockerfile"
                    def dockerImageName = "mlflow_image:latest"
                    def dockerContainerName = "mlflow_container"

                    // Detener y eliminar el contenedor si ya existe
                    sh "docker stop ${dockerContainerName} || true"
                    sh "docker rm ${dockerContainerName} || true"

                    // Crear la imagen
                    sh "docker build -t ${dockerImageName} -f ${dockerfilePath} ."

                    // Levantar el nuevo contenedor
                    sh "docker run -p 80:80  --name ${dockerContainerName} -d ${dockerImageName}"
                }
            }
        }
        stage('Docker Compose') {
            steps {
                script {
                    // Ejecuta el docker-compose
                    sh "docker-compose -f docker-compose.yaml up -d"
                }
            }
        }


        // stage('Construir imagen Docker model tree classifier') {
        //     steps {
        //         script {
        //             def dockerfilePath = "${workspaceDir}/model-tree-classifier/Dockerfile"
        //             def dockerImageName = "training_image:latest"
        //             sh "docker build -t ${dockerImageName} -f ${dockerfilePath} ."
        //         }
        //     }
        // }



    }
}
