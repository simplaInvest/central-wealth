## Objetivo
- Atualizar os dados da página "🗓️ Reuniões" a cada 15 minutos sem exigir novo login e sem tirar o usuário da visualização em tela cheia usada no escritório.

## Abordagem Técnica
- Usar `st_autorefresh` para disparar um rerun a cada 15 minutos e combinar com `@st.cache_data(ttl=900)` para que as planilhas sejam recarregadas no mesmo intervalo.
- Manter a sessão autenticada via `st.session_state.authenticated` (já presente) para não forçar novo login com os reruns.
- Evitar a perda do “fullscreen” dos gráficos substituindo o botão de fullscreen do gráfico por um "Modo Apresentação" que ocupa a janela inteira via CSS e persiste estado em `st.session_state`.

## Alterações Propostas
- `pages/2_reunioes.py:110`: ajustar `@st.cache_data(ttl=300)` para `ttl=900`.
- `pages/2_reunioes.py` (após autenticação, linhas 36–39): inserir `st_autorefresh(interval=900_000, key="reunioes_refresh")`.
- `pages/2_reunioes.py`: adicionar controles e estado do "Modo Apresentação":
  - `st.session_state.presenter_mode` (bool) e `st.session_state.presenter_view` (string) para armazenar a visão apresentada.
  - CSS que oculta header, sidebar e bordas e faz o container ocupar 100% da viewport.
  - Renderização condicional: quando `presenter_mode` estiver ativo, renderizar somente a visualização escolhida (ex.: "Funil", "Evolução Diária", "Performance SDR", "Contratos por Consultor", "Pipeline") sem as abas.
- `pages/2_reunioes.py`: exibir um pequeno status "Última atualização: HH:MM:SS" para auditoria visual em modo apresentação.
- (Opcional) URL de kiosk: ler `st.query_params` para `presenter=1&view=funil` e iniciar diretamente em modo apresentação.

## Fluxo do Usuário
- Operação normal: filtros e abas funcionam como hoje; atualização automática ocorre a cada 15 minutos sem afetar o login.
- Apresentação no escritório: ativar "Modo Apresentação" e selecionar a visualização; pressionar `F11` no navegador para fullscreen do sistema. Os reruns não sairão do modo apresentação.

## Considerações
- `st_autorefresh` causa rerun, mas com o novo "Modo Apresentação" o estado visual persiste; não há necessidade de reativar fullscreen do gráfico.
- O custo de leitura das planilhas fica contido com `ttl=900`. Se o rerun ocorrer antes do TTL expirar, nenhum acesso extra ao Google Sheets será feito.
- Segurança mantida: uso de `st.secrets` com credenciais (já existente em `pages/2_reunioes.py:67–75`).

## Validação
- Verificar que o contador do `st_autorefresh` está ativo e que `Última atualização` muda a cada ciclo.
- Confirmar que `authenticated` permanece `True` nos reruns e o acesso não é bloqueado (`pages/2_reunioes.py:36–38`).
- Testar o "Modo Apresentação" em todas as visualizações e observar se o estado se mantém após o refresh.

## Próximos Passos
1. Implementar `st_autorefresh` e ajustar `ttl`.
2. Criar o "Modo Apresentação" com estado e CSS.
3. Parametrizar o modo via `st.query_params` e acrescentar o status de última atualização.
4. Testar em ambiente local com uma aba fixa em tela cheia (F11).