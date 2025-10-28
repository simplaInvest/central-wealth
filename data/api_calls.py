import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import requests

def request_generic(url):
    try:
        resp = requests.get(url)
        resp.raise_for_status()  # dispara exceção para 4xx/5xx
        dados = resp.json()      # resposta em JSON

        return dados

    except requests.exceptions.HTTPError as e:
        print("HTTP error:", e, "| body:", getattr(e.response, "text", ""))
    except requests.exceptions.Timeout:
        print("Timeout — tente aumentar o timeout ou checar conectividade.")
    except requests.exceptions.RequestException as e:
        print("Erro de rede:", e)