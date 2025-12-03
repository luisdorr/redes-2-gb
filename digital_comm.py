"""
Simulação educativa de um sistema de transmissão digital.

Fluxo principal:
- Converte texto ASCII em bits.
- Codifica em Manchester ou Bifase-Mark (Differential Manchester).
- Modula (BPSK ou QPSK).
- Canal AWGN.
- Demodula e decodifica o sinal.
- Recupera texto e calcula a BER.

Codificadores de Canal Implementados:
- Manchester: bit 0 -> [low, high], bit 1 -> [high, low]
- BiphaseMark: Codificação diferencial com transição mid-bit sempre presente.
  bit 1 gera transição no início, bit 0 não gera transição no início.

Somente numpy e matplotlib são utilizados para manter o foco na matemática do
sinal e evitar bibliotecas "caixa preta".
"""
from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from scipy.special import erfc


class BitStreamConverter:
    """Conversão entre texto ASCII e sequências de bits."""

    @staticmethod
    def text_to_bits(text: str) -> np.ndarray:
        bytes_view = np.frombuffer(text.encode("ascii"), dtype=np.uint8)
        # np.unpackbits gera um vetor de 0s e 1s para cada byte.
        bits = np.unpackbits(bytes_view)
        return bits.astype(np.int8)

    @staticmethod
    def bits_to_text(bits: np.ndarray) -> str:
        # Garante múltiplos de 8 bits para remontar bytes.
        trimmed = bits[: len(bits) - (len(bits) % 8)]
        byte_array = np.packbits(trimmed.astype(np.uint8))
        return byte_array.tobytes().decode("ascii", errors="replace")


class ManchesterEncoder:
    """Codificação e decodificação Manchester."""

    def __init__(self, low_high: tuple[int, int] = (0, 1)) -> None:
        self.low, self.high = low_high

    def encode(self, bits: np.ndarray) -> np.ndarray:
        # Cada bit vira um par: 0 -> [low, high], 1 -> [high, low].
        encoded = np.empty(bits.size * 2, dtype=np.int8)
        encoded[0::2] = np.where(bits == 0, self.low, self.high)
        encoded[1::2] = np.where(bits == 0, self.high, self.low)
        return encoded

    def decode(self, encoded: np.ndarray) -> np.ndarray:
        if encoded.size % 2 != 0:
            raise ValueError("Sequência Manchester deve ter tamanho par.")
        pairs = encoded.reshape(-1, 2)
        decoded = np.where(
            (pairs[:, 0] == self.low) & (pairs[:, 1] == self.high),
            0,
            1,
        )
        return decoded.astype(np.int8)


class BiphaseMarkEncoder:
    """
    Codificação e decodificação Bifase-Mark (Differential Manchester).
    
    Características:
    - Sempre há transição no meio do intervalo do bit (mid-bit).
    - bit = 1: há transição no início do intervalo.
    - bit = 0: não há transição no início do intervalo.
    - Codificação diferencial que depende do nível anterior.
    """

    def __init__(self, low_high: tuple[int, int] = (0, 1), start_level: int = 1) -> None:
        self.low, self.high = low_high
        self.start_level = start_level

    def _toggle(self, level: int) -> int:
        """Alterna entre low e high."""
        return self.high if level == self.low else self.low

    def encode(self, bits: np.ndarray) -> np.ndarray:
        """
        Codifica bits em Bifase-Mark.
        Cada bit gera dois níveis (first_half, second_half).
        """
        encoded = np.empty(bits.size * 2, dtype=np.int8)
        prev_level = self.start_level
        
        for i, bit in enumerate(bits):
            # Se bit == 1, faz transição no início
            if bit == 1:
                prev_level = self._toggle(prev_level)
            
            # Primeiro half do intervalo
            first_half = prev_level
            # Transição mid-bit sempre ocorre
            second_half = self._toggle(first_half)
            
            encoded[2*i] = first_half
            encoded[2*i + 1] = second_half
            
            # O nível ao final do bit se torna o estado para o próximo
            prev_level = second_half
        
        return encoded

    def decode(self, encoded: np.ndarray) -> np.ndarray:
        """
        Decodifica sequência Bifase-Mark de volta para bits.
        Detecta transições no início de cada intervalo de bit.
        """
        if encoded.size % 2 != 0:
            raise ValueError("Sequência Bifase-Mark deve ter tamanho par.")
        
        pairs = encoded.reshape(-1, 2)
        decoded = np.empty(pairs.shape[0], dtype=np.int8)
        prev_level = self.start_level
        
        for i, (first, second) in enumerate(pairs):
            # bit = 1 se houve transição no início (first != prev_level)
            decoded[i] = 1 if first != prev_level else 0
            # Atualiza o nível anterior para o fim deste bit
            prev_level = second
        
        return decoded


class BPSKModem:
    """Modulador/Demodulador BPSK com mapeamento ±1."""

    def modulate(self, bits: np.ndarray) -> np.ndarray:
        # 0 -> -1, 1 -> +1. Resultado é real, mas usamos complexo para canal.
        symbols = 2 * bits - 1
        return symbols.astype(np.complex128)

    def demodulate(self, symbols: np.ndarray) -> np.ndarray:
        # Detecção por limiar (distância ao eixo real).
        return (symbols.real >= 0).astype(np.int8)

    bits_per_symbol = 1


class QPSKModem:
    """Modulador/Demodulador QPSK com mapeamento Gray."""

    def __init__(self) -> None:
        norm = 1 / np.sqrt(2)
        self.constellation = {
            (0, 0): norm * (1 + 1j),
            (0, 1): norm * (-1 + 1j),
            (1, 1): norm * (-1 - 1j),
            (1, 0): norm * (1 - 1j),
        }
        # Vetor de símbolos para decisão por distância euclidiana.
        self.symbol_points = np.array(list(self.constellation.values()))
        self.symbol_bits = np.array(list(self.constellation.keys()))
        self.bits_per_symbol = 2

    def modulate(self, bits: np.ndarray) -> np.ndarray:
        if len(bits) % 2 != 0:
            bits = np.pad(bits, (0, 1), constant_values=0)
        bit_pairs = bits.reshape(-1, 2)
        symbols = np.array([self.constellation[tuple(b)] for b in bit_pairs], dtype=np.complex128)
        return symbols

    def demodulate(self, symbols: np.ndarray) -> np.ndarray:
        # Calcula distâncias euclidianas vetorizadas para cada ponto da constelação.
        distances = np.abs(symbols[:, None] - self.symbol_points[None, :]) ** 2
        closest = np.argmin(distances, axis=1)
        bits_out = self.symbol_bits[closest].reshape(-1)
        return bits_out.astype(np.int8)


class AWGNChannel:
    """Canal AWGN com controle de Eb/N0 em dB."""

    def transmit(
        self, signal: np.ndarray, ebn0_db: float, bits_per_symbol: int, rng: np.random.Generator
    ) -> np.ndarray:
        signal_power = np.mean(np.abs(signal) ** 2)
        ebn0_linear = 10 ** (ebn0_db / 10)
        # Es = Eb * k -> N0 = Eb / (Eb/N0). Ruído por dimensão real: N0/2.
        noise_variance = signal_power / (2 * bits_per_symbol * ebn0_linear)
        noise_std = np.sqrt(noise_variance)
        noise = noise_std * (rng.standard_normal(signal.shape) + 1j * rng.standard_normal(signal.shape))
        return signal + noise


class DigitalTransmissionSimulator:
    """Simulador completo com codificação de canal, modulação e BER."""

    def __init__(
        self, 
        text: str, 
        ebn0_range: list[int] | np.ndarray, 
        encoder_name: str = "Manchester",
        seed: int = 0
    ) -> None:
        self.text = text
        self.ebn0_range = np.array(ebn0_range)
        self.rng = np.random.default_rng(seed)
        self.converter = BitStreamConverter()
        
        # Seleção do codificador de canal
        encoders = {
            "Manchester": ManchesterEncoder(),
            "BiphaseMark": BiphaseMarkEncoder(),
        }
        if encoder_name not in encoders:
            raise ValueError(
                f"Codificador '{encoder_name}' desconhecido. "
                f"Opções: {list(encoders.keys())}"
            )
        self.encoder = encoders[encoder_name]
        self.encoder_name = encoder_name
        
        self.channel = AWGNChannel()
        self.modems = {
            "BPSK": BPSKModem(),
            "QPSK": QPSKModem(),
        }

    def _transmit_once(self, modem_name: str, ebn0_db: float) -> tuple[int, int, str]:
        bits = self.converter.text_to_bits(self.text)
        encoded = self.encoder.encode(bits)
        modem = self.modems[modem_name]
        symbols = modem.modulate(encoded)
        noisy = self.channel.transmit(symbols, ebn0_db, modem.bits_per_symbol, self.rng)
        detected = self.modems[modem_name].demodulate(noisy)
        decoded = self.encoder.decode(detected)
        min_len = min(len(bits), len(decoded))
        bit_errors = int(np.sum(bits[:min_len] != decoded[:min_len]))
        recovered_text = self.converter.bits_to_text(decoded)
        return bit_errors, min_len, recovered_text

    def run(self) -> dict[str, list[float]]:
        ber_results: dict[str, list[float]] = {name: [] for name in self.modems}
        for ebn0_db in self.ebn0_range:
            for modem_name in self.modems:
                errors, total, _ = self._transmit_once(modem_name, ebn0_db)
                ber = errors / total if total else 0.0
                ber_results[modem_name].append(ber)
                print(
                    f"Eb/N0: {ebn0_db} dB | Modulação: {modem_name} | Erros: {errors} | BER: {ber:.6f}"
                )
        return ber_results

    def plot(self, ber_results: dict[str, list[float]], output_file: str = "ber_vs_ebn0.png") -> None:
        plt.figure(figsize=(8, 5))
        for modem_name, bers in ber_results.items():
            plt.semilogy(self.ebn0_range, bers, marker="o", linestyle="None", label=f"{modem_name} (sim)")

        # Curvas teóricas (BER por bit para BPSK e QPSK com mapeamento Gray).
        ebn0_linear = 10 ** (self.ebn0_range / 10)
        theoretical_ber = 0.5 * erfc(np.sqrt(ebn0_linear))
        plt.semilogy(self.ebn0_range, theoretical_ber, label="BPSK/QPSK (teórico)")
        plt.grid(True, which="both", linestyle="--", alpha=0.6)
        plt.xlabel("Eb/N0 (dB)")
        plt.ylabel("BER")
        plt.title(f"BER x Eb/N0 para BPSK e QPSK com codificação {self.encoder_name}")
        plt.legend()
        plt.tight_layout()
        plt.savefig(output_file, dpi=150)
        print(f"Gráfico salvo em {output_file}")


if __name__ == "__main__":
    long_text = (
        "Engenharia de Telecomunicacoes exige clareza, precisao e muita pratica para"
        " dominar os fundamentos de sistemas digitais. "
        "Esta simulacao em Python demonstra o impacto do ruido sobre diferentes"
        " modulacoes com codificacao de canal."
    )
    
    # Range estendido para capturar o comportamento em alto ruído (-10 dB)
    ebn0_values = np.arange(-10, 16, 2)
    
    # Simulação com ambos os codificadores
    encoders_to_test = ["Manchester", "BiphaseMark"]
    
    for encoder_name in encoders_to_test:
        print(f"\n{'='*60}")
        print(f"SIMULAÇÃO COM CODIFICADOR: {encoder_name}")
        print(f"{'='*60}\n")
        
        simulator = DigitalTransmissionSimulator(
            long_text, 
            ebn0_values, 
            encoder_name=encoder_name,
            seed=42
        )
        ber = simulator.run()
        
        # Salva gráfico específico para cada codificador
        output_file = f"ber_vs_ebn0_{encoder_name.lower()}.png"
        simulator.plot(ber, output_file=output_file)
        
        # Cálculo da Eficiência Espectral (mesma para ambos)
        coding_rate = 0.5  # Ambos dobram a taxa de símbolos
        efficiencies = {
            "BPSK": 1 * coding_rate,
            "QPSK": 2 * coding_rate,
        }
        print(f"\nEficiência espectral teórica (bits/s/Hz) com {encoder_name}:")
        for name, eff in efficiencies.items():
            print(f" - {name}: {eff:.2f} bits/s/Hz")
        
        # Demonstração Prática em dois níveis de SNR
        demonstration_snrs = [0, 10]  # 0 dB (ruído moderado) e 10 dB (ruído baixo)
        
        for snr in demonstration_snrs:
            print(f"\n{'='*60}")
            print(f"DEMONSTRAÇÃO PRÁTICA - {encoder_name} em Eb/N0 = {snr} dB")
            print(f"{'='*60}")
            
            for modem_name in simulator.modems:
                errors, total, recovered = simulator._transmit_once(modem_name, snr)
                ber_demo = errors / total
                print(f"\n[{modem_name}] BER: {ber_demo:.6f} (Erros: {errors}/{total})")
                print(f"Texto Recuperado:\n{recovered[:100]}...")
                print("-" * 60)
