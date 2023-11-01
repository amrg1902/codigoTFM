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

                    // Construir la ruta completa al archivo en el workspace
                    def rutaCompleta = "${workspaceDir}/${archivo}"

                    sh "cat ${rutaDockerfile}"
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
