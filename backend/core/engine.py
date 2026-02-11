import os
import torch
import bitstring
import numpy as np
from backend.core.model import Predictor
from backend.utils.arithmetic import ArithmeticEngine, Encoder, Decoder

class AICompressionEngine:
    def __init__(self, model_path=None):
        self.predictor = Predictor(model_path=model_path)
        self.engine = ArithmeticEngine(precision=32)

    def compress(self, input_path, output_path, model_id="default"):
        with open(input_path, 'rb') as f:
            data = f.read()

        orig_size = len(data)
        encoder = Encoder(self.engine)

        # Performance Optimization: Use numpy array for context to speed up tensor conversion
        # and avoid O(N) list.pop(0) operations.
        context = np.zeros(self.predictor.context_size, dtype=np.int64)
        context_len = 0

        for i, byte in enumerate(data):
            # 1. Get prediction from AI
            probs = self.predictor.predict_next_byte_dist(context[:context_len])

            # 2. Get cumulative frequencies for AC
            cum_freqs, total_count = self.engine.get_cum_freqs(probs)

            # 3. Encode byte
            encoder.encode(byte, cum_freqs, total_count)

            # 4. Update context
            if context_len < self.predictor.context_size:
                context[context_len] = byte
                context_len += 1
            else:
                # Efficient shift and update
                context[:-1] = context[1:]
                context[-1] = byte

            if i % 1000 == 0:
                print(f"Compressed {i}/{orig_size} bytes...", end='\r')

        compressed_bits = encoder.finish()

        # Save to file
        with open(output_path, 'wb') as f:
            # Header: Magic(4) + OrigSize(8) + ModelIDLen(2) + ModelID(N)
            f.write(b'AICP')
            f.write(orig_size.to_bytes(8, 'big'))
            model_id_bytes = model_id.encode('utf-8')
            f.write(len(model_id_bytes).to_bytes(2, 'big'))
            f.write(model_id_bytes)

            # Write bits
            b = bitstring.BitArray(compressed_bits)
            f.write(b.tobytes())

        print(f"\nCompression complete: {orig_size} -> {os.path.getsize(output_path)} bytes")

    def decompress(self, input_path, output_path, model_library_path="models"):
        with open(input_path, 'rb') as f:
            magic = f.read(4)
            if magic != b'AICP':
                raise ValueError("Not a valid AICP file")

            orig_size = int.from_bytes(f.read(8), 'big')
            model_id_len = int.from_bytes(f.read(2), 'big')
            model_id = f.read(model_id_len).decode('utf-8')

            # Load the correct model if it's not the current one
            model_file = os.path.join(model_library_path, f"{model_id}.pt")
            if os.path.exists(model_file):
                self.predictor.load(model_file)

            data_bits = bitstring.BitArray(f.read())

        decoder = Decoder(self.engine, data_bits)

        # Performance Optimization: Use numpy array for context to speed up tensor conversion
        context = np.zeros(self.predictor.context_size, dtype=np.int64)
        context_len = 0
        decoded_data = bytearray()

        for i in range(orig_size):
            probs = self.predictor.predict_next_byte_dist(context[:context_len])
            cum_freqs, total_count = self.engine.get_cum_freqs(probs)

            byte = decoder.decode(cum_freqs, total_count)
            decoded_data.append(byte)

            # Update context
            if context_len < self.predictor.context_size:
                context[context_len] = byte
                context_len += 1
            else:
                # Efficient shift and update
                context[:-1] = context[1:]
                context[-1] = byte

            if i % 1000 == 0:
                print(f"Decompressed {i}/{orig_size} bytes...", end='\r')

        with open(output_path, 'wb') as f:
            f.write(decoded_data)

        print(f"\nDecompression complete.")
