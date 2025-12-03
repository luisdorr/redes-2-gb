# Simulação de Transmissão Digital (BPSK e QPSK)

Projeto educacional em Python que demonstra todas as etapas de um enlace digital:
conversão de texto em bits, codificação de canal (Manchester ou Bifase-Mark), 
modulação (BPSK e QPSK), canal AWGN, detecção, decodificação e cálculo de BER.

## Requisitos
- Python 3.10+
- numpy
- matplotlib
- scipy

## Como executar
1. Instale as dependências (se necessário):
   ```bash
   pip install numpy matplotlib scipy
   ```
2. Rode a simulação padrão que testa ambos os codificadores (Manchester e BiphaseMark):
   ```bash
   python3 digital_comm.py
   ```

A execução imprime logs no console com o número de erros e BER para cada Eb/N0 e
modulação, e gera dois gráficos:
- `ber_vs_ebn0_manchester.png` - Resultados com codificador Manchester
- `ber_vs_ebn0_biphasemark.png` - Resultados com codificador Bifase-Mark

### Uso Programático
Você pode usar o simulador com diferentes configurações:

```python
from digital_comm import DigitalTransmissionSimulator
import numpy as np

# Simular apenas com Manchester
simulator = DigitalTransmissionSimulator(
    text="Mensagem de teste",
    ebn0_range=np.arange(0, 15, 2),
    encoder_name="Manchester",  # ou "BiphaseMark"
    seed=42
)
ber_results = simulator.run()
simulator.plot(ber_results, output_file="meu_grafico.png")
```

## Organização do código
- `BitStreamConverter`: converte texto ASCII para bits (via `numpy.unpackbits`) e
  bits de volta para texto.
- `ManchesterEncoder`: aplica a codificação Manchester (0 → [0, 1], 1 → [1, 0])
  e sua decodificação inversa.
- `BiphaseMarkEncoder`: aplica a codificação Bifase-Mark (Differential Manchester).
  Codificação diferencial onde:
  - Sempre há transição no meio do bit (mid-bit)
  - bit 1: há transição no início do intervalo
  - bit 0: não há transição no início do intervalo
- `BPSKModem` e `QPSKModem`: moduladores/demoduladores baseados em distância
  euclidiana mínima. O QPSK usa mapeamento Gray normalizado por \(1/\sqrt{2}\).
- `AWGNChannel`: adiciona ruído branco gaussiano aditivo a partir do Eb/N0 em dB.
- `DigitalTransmissionSimulator`: orquestra uma transmissão completa, calcula
  BER e gera o gráfico comparativo. Suporta seleção do codificador de canal.

## Observações teóricas

### Modulação
- **Eficiência espectral**: QPSK transmite 2 bits por símbolo, dobrando a
  eficiência espectral em relação ao BPSK para a mesma largura de banda.
- **Robustez a ruído**: BPSK possui maior distância mínima entre símbolos,
  apresentando BER levemente menor que QPSK para a mesma relação Eb/N0.

### Codificadores de Canal
- **Manchester**: Codificação não-diferencial onde 0 → [low, high] e 1 → [high, low].
  Dobra a taxa de símbolos, mas fornece componente DC nula e facilita a 
  sincronização de relógio.
- **Bifase-Mark (Differential Manchester)**: Codificação diferencial com transição 
  mid-bit sempre presente. O bit é determinado pela presença (bit 1) ou ausência 
  (bit 0) de transição no início do intervalo. Oferece vantagens similares ao 
  Manchester com maior robustez a inversões de polaridade.

### Comparação entre Codificadores
Ambos os codificadores (Manchester e Bifase-Mark) possuem:
- Taxa de codificação de 0.5 (dobram a quantidade de símbolos)
- Eficiência espectral: BPSK = 0.5 bits/s/Hz, QPSK = 1.0 bits/s/Hz
- Desempenho BER similar em condições ideais
- Componente DC nula e auto-sincronização

O código está amplamente comentado para destacar o cálculo de variância do
ruído, mapeamento de constelações e etapas de codificação/decodificação.
