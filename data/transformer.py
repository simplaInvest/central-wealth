def formata_reais(valor: float) -> str:
    """
    Converte um número (float ou int) em formato monetário brasileiro.

    Exemplo:
        738873539.01 → 'R$ 738.873.539,01'
    """
    if not isinstance(valor, (int, float)):
        raise ValueError("O valor deve ser numérico (int ou float).")
    return f"R$ {valor:,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")

def formata_milhoes_brl(valor: float) -> str:
    if valor is None:
        return "R$ 0,00 M"
    # transforma 123456789.01 -> "R$ 123,46 M" (pt-BR: vírgula decimal)
    s = f"{valor/1_000_000:,.2f}"            # usa , como separador milhar (padrão US)
    s = s.replace(",", "X").replace(".", ",").replace("X", ".")  # troca para pt-BR
    return f"R$ {s} M"

def formata_delta_milhoes_brl(valor: float) -> str:
    sinal = "+" if valor >= 0 else "-"
    abs_val = abs(valor)
    s = f"{abs_val/1_000_000:,.2f}"
    s = s.replace(",", "X").replace(".", ",").replace("X", ".")
    return f"{sinal} R$ {s} M"
