conda create -n layoutlm python=3.7
conda activate layoutlm
conda init
conda install pytorch==1.4.0 cudatoolkit=10.1 -c pytorch
pip install -r requirements.txt

cd model
git clone https://github.com/microsoft/unilm.git
cd unilm/layoutlm/deprecated
pip install .

pip install --upgrade "protobuf<=3.20.1"