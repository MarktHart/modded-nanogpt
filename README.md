# Start hier
1. Maak een python venv aan met `python -m venv env`
2. Activeer de env met `./env/bin/activate`
3. Installeer de requirements met `pip install -r requirements.txt`
4. Download de data met `python data/cached_fineweb10B.py 1`
5. Voer een run uit met `torchrun --standalone --nproc_per_node=1 demo_gpt.py`
