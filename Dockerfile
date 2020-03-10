FROM nvidia/cuda:10.0-base-ubuntu18.04

# Update apt
RUN apt-get -y update

# Install python
RUN apt-get install -y python3.7 \
    python3-pip

# Upgrade pip
RUN pip3 install --no-cache-dir --upgrade pip 

# Install your python libraries
RUN pip3 install --no-cache-dir numpy \
    matplotlib \
    scikit-learn \ 
    pillow \
    requests \
    flask \
    tqdm \
    tensorflow-gpu \
    keras

# Copy your code
COPY src src

# # Command to run app.py
WORKDIR src
CMD python3 app.py

# # Expose port
EXPOSE 5000