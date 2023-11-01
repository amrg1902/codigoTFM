pipeline {
    agent any

    stages {
        stage('Mensaje de Éxito') {
            steps {
                echo '¡Clonado exitosamente desde GitHub!'
            }
        }
    }
}
