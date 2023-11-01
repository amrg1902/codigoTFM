pipeline {
    agent any

    stages {
        stage('Install MLflow') {
            steps {
                sh '/Users/aitormartin-romogonzalez/opt/anaconda3/bin/python -m venv venv'
                sh 'source venv/bin/activate'
                sh '/Users/aitormartin-romogonzalez/opt/anaconda3/bin/pip install mlflow'
            }
        }


        stage('Run MLflow Server') {
            steps {
                sh 'mlflow server --host 0.0.0.0 --port 5000'
            }
        }
    }
}
