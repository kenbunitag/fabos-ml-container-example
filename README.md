# fabos-ml-container-example
Example container that serves a pretrained ML Models with a simple flask-based webserver.


## Option 1: Run as python app
```
# Install pytroch
conda install pytorch torchvision torchaudio cudatoolkit=10.2 -c pytorch

# install the rest
pip install -r requirements.py

# start
python app.py

# open browser at http://localhost:5000
```

## Option 2: Run in Docker
```
docker-compose up

# open browser at http://localhost:5000
```
