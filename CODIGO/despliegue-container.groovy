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
                    if (path.contains('/Users/aitormartin-romogonzalez/opt/anaconda3/bin')) {
                        echo "La ruta /Users/aitormartin-romogonzalez/opt/anaconda3/bin está presente en PATH."
                    } else {
                        error "La ruta /Users/aitormartin-romogonzalez/opt/anaconda3/bin no está presente en PATH."
                    }
                }
            }
        }
    }
}
