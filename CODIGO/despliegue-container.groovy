pipeline {
    agent any

    stages {
        stage('Ejemplo de Acceso al Workspace') {
            steps {
                script {
                    // Obtener la ruta al directorio de trabajo (workspace)
                    def workspaceDir = env.WORKSPACE

                    // Definir el nombre del archivo que deseas acceder
                    def archivo = "${workspaceDir}/CODIGO/Dockerfile"

                    // Mostrar un mensaje en el registro de Jenkins
                    echo "Contenido del archivo ${archivo}:"
                    
                    // Utilizar 'echo' para mostrar el contenido del archivo
                    sh "echo \"\$(cat ${archivo})\""
                }
            }
        }
        stage('Mensaje de Éxito') {
            steps {
                echo '¡Clonado exitosamente desde GitHub!'
            }
        }
    }
}
