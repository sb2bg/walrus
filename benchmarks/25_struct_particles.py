# Struct/Object Benchmark - allocation, field mutation, and instance calls


class Particle:
    def __init__(self, x, y, vx, vy):
        self.x = x
        self.y = y
        self.vx = vx
        self.vy = vy

    def step(self, bias):
        self.x = self.x + self.vx + bias
        self.y = self.y + self.vy + (bias % 3)
        self.vx = self.vx + ((bias + 1) % 3) - 1
        self.vy = self.vy + ((bias + 2) % 5) - 2
        return self.x + self.y + self.vx + self.vy


def build_particles(n):
    particles = []
    for i in range(n):
        particles.append(Particle(i, i * 2, i % 7, (i * 3) % 11))
    return particles


def particle_benchmark(count, steps):
    particles = build_particles(count)
    total = 0
    for step in range(steps):
        bias = step % 5
        for i in range(len(particles)):
            total += particles[i].step(bias)
    return total


result = particle_benchmark(600, 200)

print(f"Struct particle workload: {result}")
