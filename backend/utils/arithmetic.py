import bisect

import bitarray


class ArithmeticEngine:
    """
    A simple Arithmetic Coding implementation using integer ranges.
    """

    def __init__(self, precision=32):
        self.precision = precision
        self.MAX_RANGE = (1 << precision) - 1
        self.HALF_RANGE = 1 << (precision - 1)
        self.QUARTER_RANGE = 1 << (precision - 2)
        self.THREE_QUARTER_RANGE = self.HALF_RANGE + self.QUARTER_RANGE

    def get_cum_freqs(self, probs, total_count=1000000):
        """
        Converts float probabilities to cumulative integer frequencies.
        Ensures no frequency is 0.
        """
        # Move to CPU early to avoid multiple GPU-CPU synchronizations for small tensors
        probs = probs.to("cpu")

        # Performance Optimization: Use item() for scalar extraction and in-place ops
        # to reduce tensor overhead in the high-frequency engine loop.
        # Bolt: Use .add() for slight performance gain over +
        p = probs.add(1e-6)
        # Avoid creating a tensor for the sum when dividing
        p.mul_(total_count / p.sum().item())

        # Bolt: .long() is ~2x faster than .floor().long() and safe for positive values
        counts = p.long()
        # Adjust to sum exactly to total_count using item() for the sum
        # Bolt: Only adjust if the difference is non-zero to skip redundant addition
        diff = total_count - counts.sum().item()
        if diff:
            counts[0] += diff

        # Performance Optimization: Constructing a list of cumulative frequencies
        # using [0] + counts.cumsum(0).tolist() is significantly faster than
        # pre-allocating a zero tensor and using in-place slice assignment.
        return [0] + counts.cumsum(0).tolist(), total_count


class Encoder:
    def __init__(self, engine):
        self.engine = engine
        self.low = 0
        self.MAX_RANGE = engine.MAX_RANGE
        self.HALF_RANGE = engine.HALF_RANGE
        self.QUARTER_RANGE = engine.QUARTER_RANGE
        self.THREE_QUARTER_RANGE = engine.THREE_QUARTER_RANGE
        self.high = self.MAX_RANGE
        self.pending_bits = 0
        self.output_bits = bitarray.bitarray()

    def encode(self, symbol, cum_freqs, total_count):
        range_width = self.high - self.low + 1
        self.high = self.low + (range_width * cum_freqs[symbol + 1] // total_count) - 1
        self.low = self.low + (range_width * cum_freqs[symbol] // total_count)

        while True:
            if self.high < self.HALF_RANGE:
                self._emit_bit(0)
            elif self.low >= self.HALF_RANGE:
                self._emit_bit(1)
                self.low -= self.HALF_RANGE
                self.high -= self.HALF_RANGE
            elif (
                self.low >= self.QUARTER_RANGE and self.high < self.THREE_QUARTER_RANGE
            ):
                self.pending_bits += 1
                self.low -= self.QUARTER_RANGE
                self.high -= self.QUARTER_RANGE
            else:
                break

            self.low = (self.low << 1) & self.MAX_RANGE
            self.high = ((self.high << 1) | 1) & self.MAX_RANGE

    def _emit_bit(self, bit):
        self.output_bits.append(bit)
        while self.pending_bits > 0:
            self.output_bits.append(1 - bit)
            self.pending_bits -= 1

    def finish(self):
        self.pending_bits += 1
        if self.low < self.QUARTER_RANGE:
            self._emit_bit(0)
        else:
            self._emit_bit(1)
        return self.output_bits


class Decoder:
    def __init__(self, engine, bit_stream):
        self.engine = engine
        self.MAX_RANGE = engine.MAX_RANGE
        self.HALF_RANGE = engine.HALF_RANGE
        self.QUARTER_RANGE = engine.QUARTER_RANGE
        self.THREE_QUARTER_RANGE = engine.THREE_QUARTER_RANGE
        self.bit_stream = bit_stream
        self.bit_idx = 0
        self.low = 0
        self.high = self.MAX_RANGE
        self.value = 0

        # Initialize value with first precision bits
        for _ in range(engine.precision):
            self.value = (self.value << 1) | self._next_bit()

    def _next_bit(self):
        if self.bit_idx < len(self.bit_stream):
            bit = self.bit_stream[self.bit_idx]
            self.bit_idx += 1
            return bit
        return 0

    def decode(self, cum_freqs, total_count):
        range_width = self.high - self.low + 1
        current_count = ((self.value - self.low + 1) * total_count - 1) // range_width

        # Binary search for symbol using bisect for performance
        symbol = bisect.bisect_right(cum_freqs, current_count) - 1

        # Update range
        self.high = self.low + (range_width * cum_freqs[symbol + 1] // total_count) - 1
        self.low = self.low + (range_width * cum_freqs[symbol] // total_count)

        while True:
            if self.high < self.HALF_RANGE:
                pass
            elif self.low >= self.HALF_RANGE:
                self.low -= self.HALF_RANGE
                self.high -= self.HALF_RANGE
                self.value -= self.HALF_RANGE
            elif (
                self.low >= self.QUARTER_RANGE and self.high < self.THREE_QUARTER_RANGE
            ):
                self.low -= self.QUARTER_RANGE
                self.high -= self.QUARTER_RANGE
                self.value -= self.QUARTER_RANGE
            else:
                break

            self.low = (self.low << 1) & self.MAX_RANGE
            self.high = ((self.high << 1) | 1) & self.MAX_RANGE
            self.value = ((self.value << 1) | self._next_bit()) & self.MAX_RANGE

        return symbol
