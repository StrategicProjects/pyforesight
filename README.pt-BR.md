# pyforesight

Previsão de séries temporais em Python que escolhe o modelo pelo que teria
dado certo.

O `foresight` ajusta vários modelos a uma série, refaz o passado para ver como
cada um teria se saído, escolhe pelo erro fora da amostra e dá faixas tiradas
dos erros de fato observados, inclusive para o total dos próximos k períodos.

Os modelos, o backtest e as ferramentas são o crate Rust
[foresight](https://github.com/milkway/foresight), compilado dentro do pacote:
os números são os do crate, o backtest usa todos os núcleos e nada mais é
preciso para rodar. Aceita NumPy e pandas, e devolve pandas quando pedido,
mas nenhum dos dois é obrigatório.

<p align="center"><img src="https://raw.githubusercontent.com/StrategicProjects/pyforesight/main/docs/figures/architecture.svg" alt="Arquitetura do foresight" width="100%"></p>

**Site:** <https://strategicprojects.github.io/pyforesight/>

## Instalação

```bash
pip install pyforesight
```

Python 3.9 ou mais novo. Há wheels prontos para Linux, macOS e Windows; o
pacote é importado como `foresight`. DOI: <https://doi.org/10.5281/zenodo.23050402>.

## Uso

```python
import foresight as fs

y = fs.monthly(valores, first_month=3)   # primeira observação em março
relatorio = fs.backtest(y)               # 36 origens, 12 meses à frente, 11 modelos
melhor = relatorio.best
melhor.name, melhor.score, melhor.forecast[0].interval(0.80)
melhor.cumulative(6)                     # total dos próximos seis meses, com faixa própria
relatorio.to_pandas()
```

## O que tem

- Modelos: média, ingênuo, tendência, sazonal ingênuo, Theta, Holt-Winters,
  regressão log-linear (com deflator opcional), ARIMA sazonal por máxima
  verossimilhança exata (com regressores), ARIMA automático, família ETS com
  escolha automática, Prophet (quebras de tendência, eventos e degraus), TBATS
  e Croston, SBA e TSB para demanda intermitente.
- Decomposição STL e MSTL, e qualquer modelo sobre a série dessazonalizada.
- Ensemble: média, mediana, pesos pelo inverso do erro ou pesos empilhados.
- Limpeza: preenchimento de falhas e troca de valores atípicos.
- Backtest por origem móvel em todos os núcleos, com MAPE, MAE, RMSE, MASE e
  viés por horizonte e faixas empíricas.

Os números são os do crate Rust, conferido com os pacotes `forecast` e
`prophet` do R e reproduzido de forma independente pela edição em Go.

## Autores

André Leite, Marcos Wasiliew, Hugo Vasconcelos, Carlos Amorim e Diogo Bezerra.

## Licença

MIT.
