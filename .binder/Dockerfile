FROM ghcr.io/alicepeyraut/uncertainty-quantification-paper-demos-github:latest

# Copy repo into the image, cf. https://mybinder.readthedocs.io/en/latest/tutorials/dockerfile.html. MG20230531: OMG this copies from the "build context", cf. https://stackoverflow.com/questions/73156067/where-does-the-copy-command-in-docker-copy-from; here it seems to be the repo itself.
ARG NB_USER=jovyan
ARG NB_UID=1000
USER ${NB_USER}
ENV HOME /home/${NB_USER}
COPY --chown=${NB_UID} . ${HOME}
