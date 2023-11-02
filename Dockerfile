FROM ubuntu:22.04 
#Utilidades
RUN apt−get RUN apt−get RUN apt−get
RUN apt−get update
RUN apt−get install −y wget install −y libpq−dev
#Python
RUN apt−get install −y install python3 
# Usuario sin privilegios
# −−login: Fuerza a la ejecucion de los archivos de configuración
# −c Permite que todos los argumentos vengan dentro del mismo string # y los separa
SHELL ["/bin/bash", "−−login", "−c"]
ENV USER=devteam
ENV HOME=/home/${USER}
# Recomendadión para designar usuarios sin privilegios (UID, GID) 
ENV UID 1000
ENV GID 100
# Crea el usuario sin contraseña ,
# fuera de la lista de sudoers con los parametros anteriormente
# definidos
RUN adduser −−disabled−password \ 
    −−gecos "Non−root user" \ −−home $ {HOME} \
    −−uid ${UID} \
    −−gid ${GID} \
    ${USER}
# Copiar ficheros necesarios 
COPY entrypoint.sh ${HOME}
RUN chown ${UID}:${GID} ${HOME}/entrypoint.sh && \ chmod 0755 ${HOME}/entrypoint.sh
COPY .bashrc ${HOME}
RUN chown ${UID}:${GID} ${HOME}/.bashrc 
RUN mkdir /home/MLFLOWSRUN 
RUN chmod 0770 /home/MLFLOWS
RUN chown devteam:users /home/MLFLOWS
# Conda
USER ${USER}
ENV CONDA_DIR=${HOME}/conda 
RUN wget −−quiet \https://repo.anaconda.com/miniconda/Miniconda3−latest−Linux−x86_64.sh −O ~/miniconda.sh
RUN chmod 0700 ~/miniconda.sh
RUN bash ~/miniconda.sh −b −p ${CONDA_DIR}
RUN rm ~/miniconda.sh
ENV PATH=$CONDA_DIR/bin:$PATH
RUN echo ". ${CONDA_DIR}/etc/profile.d/conda.sh" >> ~/.profile RUN conda init bash
#Entorno
RUN mkdir ${HOME}/artifacts
RUN mkdir ${HOME}/mlruns
ENV PROYECTOS_DIR ${HOME}/mlflowProjects RUN mkdir ${PROYECTOS_DIR}
WORKDIR ${PROYECTOS_DIR}
#Configuarar conda
ENV ENV_PREFIX=mlflow_env
RUN conda update −−name base -−channel defaults
RUN conda create −n $ENV_PREFIX
RUN conda clean −−all −−yes
RUN conda activate ${ENV_PREFIX} && \python3 −m pip install −−upgrade pip && \ pip install mlflow && \pip install psycopg2−binary && \pip install hdfs
ENTRYPOINT ["/home/devteam/entrypoint.sh"] # Lanzamiento servidor
EXPOSE 80