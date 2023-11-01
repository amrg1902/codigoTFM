pipeline {
    agent any

    stages {
        stage('Check PATH') {
            steps {
                script {
                    // Obtenemos el valor de la variable PATH
                    def path = sh(script: 'echo $PATH', returnStdout: true).trim()
                    
                    // Imprimimos el contenido de la variable PATH
                    echo "Contenido de la variable PATH:"
                    echo path
                    
                    // Puedes agregar condiciones o comprobar valores específicos en la variable PATH aquí
                    // Por ejemplo, verificar si una ruta específica está presente
                    if (path.contains('/ruta/especifica')) {
                        echo "La ruta /ruta/especifica está presente en PATH."
                    } else {
                        error "La ruta /ruta/especifica no está presente en PATH."
                    }
                }
            }
        }
    }
}
