# Use the official Miniconda3 image as a parent image.
FROM continuumio/miniconda3

# Clone the class repository.
RUN apt-get update && apt-get install -y git curl
RUN git clone https://github.com/drc-cs/SPRING25-DATA-INTENSIVE-SYSTEMS.git

# Set the working directory.
WORKDIR /SPRING25-DATA-INTENSIVE-SYSTEMS

# Create a new Conda environment from the environment.yml file.
RUN conda env create -f environment.yml

# Activate the new environment.
RUN echo "conda activate mbai-dis" >> ~/.bashrc
SHELL ["/bin/bash", "-c"]
RUN source activate mbai-dis

# Install vscode server.
RUN curl -fsSL https://code-server.dev/install.sh | bash

# Add code-server to PATH
ENV PATH="/root/.local/bin:${PATH}"

# Expose the port that the server is running on.
EXPOSE 8080

# Run the code-server command when the container starts.
CMD ["code-server", "--auth", "none", "--bind-addr", "0.0.0.0:8080", "."]