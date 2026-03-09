# Unicode String Benchmark - unicode indexing plus ASCII substring slicing


UNICODE_TEXT = "🙂walrusédataßλ漢字" * 8
ASCII_TEXT = "walrus-benchmark-runtime-surface-" * 4


def string_benchmark(unicode_text, ascii_text, rounds):
    total = 0
    unicode_limit = 120
    ascii_limit = len(ascii_text) - 6
    for i in range(rounds):
        idx = (i * 13 + 5) % unicode_limit
        ch = unicode_text[idx]
        if ch == "🙂":
            total += 5
        elif ch == "é":
            total += 7
        elif ch == "漢":
            total += 11
        else:
            total += 1

        piece_from = (i * 11 + 3) % ascii_limit
        piece = ascii_text[piece_from : piece_from + 6]
        total += len(piece)
    return total


result = string_benchmark(UNICODE_TEXT, ASCII_TEXT, 50000)

print(f"Unicode string workload: {result}")
