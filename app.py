import streamlit as st

params = st.query_params
auth = params.get("auth", None)
page = params.get("page", None)

if auth and "authenticated" not in st.session_state:
    st.session_state.authenticated = True
    st.session_state.user_type = auth

    # Redireciona se a página for passada na URL
    if page == "metrics":
        st.switch_page("pages/1_central.py")

# Carrega as senhas do secrets.toml
logins = st.secrets["logins"]

# Configuração inicial
st.set_page_config(page_title="Login - Expert Comercial Wealth", 
                   page_icon="📊", 
                   layout="wide", 
                   initial_sidebar_state="collapsed")

# Inicializa o estado de login
if "authenticated" not in st.session_state:
    st.session_state.authenticated = False
    st.session_state.user_type = None
    st.session_state.team = None

# Oculta a sidebar completamente para quem não está logado
if not st.session_state.authenticated:
    st.sidebar.empty()
    st.markdown(
        """
        <style>
        /* Esconde o container da sidebar */
        [data-testid="stSidebar"] { display: none !important; }
        /* Opcional: evita qualquer espaço reservado */
        .css-1d391kg { display: none !important; }
        </style>
        """,
        unsafe_allow_html=True,
    )

# Página de Login
def main():
    # Centraliza a logo
    col1, col2, col3 = st.columns([1,2,1])
    with col2:
        
        st.markdown("<br>", unsafe_allow_html=True)

        st.image("assets/z_logo_light.png", use_container_width=True) # Substitua pelo caminho correto da logo
    
        st.markdown("<br>", unsafe_allow_html=True)

        # Formulário de Login'
        with st.form("login_form"):
            st.subheader("Acesso ao Expert Comercial Wealth:")
            username = st.text_input("Nome", placeholder="Digite seu nome")
            password = st.text_input("Senha", type="password", placeholder="Digite sua senha")
            login_button = st.form_submit_button("Entrar")
            
    # Verifica o login e define permissões
    if login_button:
        if username == "admin" and password == logins["admin"]:
            st.session_state.authenticated = True
            st.session_state.user_type = "admin"
            st.query_params.update(auth="admin")
            st.query_params.update(auth="admin", page="metrics")
            st.switch_page("pages/1_central.py")
        else:
            st.session_state.authenticated = False
            st.session_state.user_type = None
            st.error("Nome ou senha incorretos. Tente novamente.")



# Executa a função principal
if __name__ == "__main__":
    main()
