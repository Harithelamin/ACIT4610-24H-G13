import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Liste over aksje-tickere
stock_tickers = [
    'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 
    'META', 'NFLX', 'NVDA', 'BRK-B', 'JNJ', 
    'JPM', 'V', 'PG', 'DIS', 'ADBE', 
    'PYPL', 'INTC', 'CSCO', 'XOM', 'WMT'
]

# Hent aksjedata for perioden 1. januar 2018 til 31. desember 2022
stock_data = yf.download(stock_tickers, start='2018-01-01', end='2022-12-31')

# 1. Beregn månedlig avkastning
# Bruk 'Close' prisene og beregn månedlig avkastning
monthly_returns = stock_data['Close'].pct_change().resample('M').sum()

# 2. Beregn kovariansmatrisen
cov_matrix = monthly_returns.cov()

# 3. Visualiser kovariansmatrisen
plt.figure(figsize=(12, 8))
plt.imshow(cov_matrix, cmap='coolwarm', interpolation='none')
plt.colorbar()
plt.title("Kovariansmatrise mellom aksjene (2018-2022)")
plt.xticks(range(len(stock_tickers)), stock_tickers, rotation=90)
plt.yticks(range(len(stock_tickers)), stock_tickers)
plt.tight_layout()
plt.show()

# 4. Gjennomsnittlig månedlig avkastning for hver aksje
average_monthly_returns = monthly_returns.mean()

# Visualiser de gjennomsnittlige månedlige avkastningene
plt.figure(figsize=(10, 6))
average_monthly_returns.plot(kind='bar', color='skyblue')
plt.title('Gjennomsnittlig månedlig avkastning (2018-2022)')
plt.ylabel('Månedlig avkastning (%)')
plt.xlabel('Aksje')
plt.xticks(rotation=45)
plt.show()

# 5. Kumulativ avkastning for hver aksje
cumulative_returns = (1 + monthly_returns).cumprod() - 1

# Visualiser den kumulative avkastningen for aksjene
plt.figure(figsize=(12, 8))
cumulative_returns.plot(kind='line', marker='o')
plt.title('Kumulativ avkastning for aksjene (2018-2022)')
plt.ylabel('Kumulativ avkastning (%)')
plt.xlabel('Tid')
plt.legend(stock_tickers, loc='upper left', bbox_to_anchor=(1, 1))
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# 6. Skriv ut de siste månedlige avkastningene og kovariansmatrisen
print("Siste månedlige avkastninger for aksjene:")
print(monthly_returns.tail())

print("\nKovariansmatrisen for aksjene:")
print(cov_matrix)

print("\nGjennomsnittlig månedlig avkastning for alle aksjer:")
print(average_monthly_returns)

import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Parameters for Evolutionary Programming
POPULATION_SIZE1 = 50  # 1. Population size should be defined
OFFSPRING_SIZE = 100  # 3. Offspring size should be defined
MUTATION_RATE = 0.05  # 5. Mutation rates should be defined
GENERATIONS = 200     # 7. Number of generations should be defined
NUM_RUNS = 10         # Number of runs to assess consistency

# Define stock tickers and fetch data
stock_tickers = [
    'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 
    'META', 'NFLX', 'NVDA', 'BRK-B', 'JNJ', 
    'JPM', 'V', 'PG', 'DIS', 'ADBE', 
    'PYPL', 'INTC', 'CSCO', 'XOM', 'WMT'
]
stock_data = yf.download(stock_tickers, start='2018-01-01', end='2022-12-31')
monthly_returns = stock_data['Close'].pct_change().resample('M').sum()
cov_matrix = monthly_returns.cov()
average_monthly_returns = monthly_returns.mean()

# Evolutionary Programming Functions
def initialize_population(size, num_assets):
    return np.random.dirichlet(np.ones(num_assets), size=size)

def fitness(weights, returns, cov_matrix):
    portfolio_return = np.dot(weights, returns)
    portfolio_variance = np.dot(weights.T, np.dot(cov_matrix, weights))
    return portfolio_return / np.sqrt(portfolio_variance)  # Sharpe ratio

def mutate(weights, mutation_rate):
    for i in range(len(weights)):
        if np.random.rand() < mutation_rate:
            weights[i] += np.random.normal(0, 0.1)
    weights = np.abs(weights)  # Ensure weights remain positive
    return weights / np.sum(weights)  # Normalize weights

# Run Evolutionary Programming
def run_evolutionary_programming():
    population = initialize_population(POPULATION_SIZE1, len(stock_tickers))
    best_returns = []
    for gen in range(GENERATIONS):
        offspring = []
        for _ in range(OFFSPRING_SIZE):
            parent = population[np.random.randint(0, POPULATION_SIZE1)]
            child = mutate(parent, MUTATION_RATE)
            offspring.append(child)
        population = np.vstack([population, offspring])
        population_fitness = [fitness(ind, average_monthly_returns, cov_matrix) for ind in population]
        best_indices = np.argsort(population_fitness)[-POPULATION_SIZE1:]
        population = population[best_indices]
        best_returns.append(population_fitness[best_indices[-1]])
    
    return best_returns, population[np.argmax(population_fitness)]

# Measure expected return and convergence for multiple runs
all_best_returns = []
for run in range(NUM_RUNS):
    best_returns, best_weights = run_evolutionary_programming()
    all_best_returns.append(best_returns)

# Plot expected return of the optimized portfolio
plt.figure(figsize=(10, 6))
plt.plot(best_returns, label="Expected Return")
plt.title('Expected Return of Optimized Portfolio')
plt.xlabel('Generation')
plt.ylabel('Expected Return')
plt.legend()
plt.show()

# Plot convergence across multiple runs
for run, best_returns in enumerate(all_best_returns):
    plt.plot(best_returns, label=f"Run {run+1}")
plt.title('Convergence to Near-Optimal Solution')
plt.xlabel('Generation')
plt.ylabel('Expected Return')
plt.legend()
plt.show()

# Plot consistency across runs
consistency = np.mean(all_best_returns, axis=0)
plt.figure(figsize=(10, 6))
plt.plot(consistency, label="Mean Expected Return Across Runs")
plt.fill_between(range(GENERATIONS), 
                 consistency - np.std(all_best_returns, axis=0), 
                 consistency + np.std(all_best_returns, axis=0), 
                 alpha=0.2, color="gray", label="Std. Dev.")
plt.title('Consistency of Results Across Runs')
plt.xlabel('Generation')
plt.ylabel('Expected Return')
plt.legend()
plt.show()
# Create CSV file for results
results_df = pd.DataFrame({
    'Generation': range(GENERATIONS),
    'Mean Expected Return': consistency
})
results_df.to_csv("ep_optimization_results.csv", index=False)
#---------------------
#---------------------
#---------------------
#---------------------
#---------------------2
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

# Set parameters for Advanced Evolutionary Programming
POPULATION_SIZE2 = 100          # Define population size
OFFSPRING_SIZE = 100          # Define offspring size
INITIAL_MUTATION_RATE = 0.05  # Define initial mutation rate
GENERATIONS = 200             # Define number of generations
ELITISM_RATE = 0.1            # Proportion of top individuals to retain as elite
NUM_RUNS = 10                 # Number of runs to assess consistency
OUTPUT_FOLDER = "ep_results_v2"
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# Define stock tickers and fetch data
stock_tickers = [
    'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 
    'META', 'NFLX', 'NVDA', 'BRK-B', 'JNJ', 
    'JPM', 'V', 'PG', 'DIS', 'ADBE', 
    'PYPL', 'INTC', 'CSCO', 'XOM', 'WMT'
]
stock_data = yf.download(stock_tickers, start='2018-01-01', end='2022-12-31')
monthly_returns = stock_data['Close'].pct_change().resample('M').sum()
cov_matrix = monthly_returns.cov()
average_monthly_returns = monthly_returns.mean()

# Advanced EP Functions
def initialize_population(size, num_assets):
    population = np.random.dirichlet(np.ones(num_assets), size=size)
    mutation_rates = np.full(size, INITIAL_MUTATION_RATE)  # Each individual has a self-adaptive mutation rate
    return population, mutation_rates

def fitness(weights, returns, cov_matrix):
    portfolio_return = np.dot(weights, returns)
    portfolio_variance = np.dot(weights.T, np.dot(cov_matrix, weights))
    return portfolio_return / np.sqrt(portfolio_variance)  # Sharpe ratio

def mutate(weights, mutation_rate):
    noise = np.random.normal(0, mutation_rate, len(weights))
    mutated_weights = weights + noise
    mutated_weights = np.abs(mutated_weights)  # Ensure weights remain positive
    return mutated_weights / np.sum(mutated_weights)  # Normalize weights

def self_adaptive_mutation(mutation_rate):
    tau = 1 / np.sqrt(len(stock_tickers))
    return mutation_rate * np.exp(tau * np.random.normal())

# Run Advanced Evolutionary Programming
def run_advanced_ep():
    population, mutation_rates = initialize_population(POPULATION_SIZE2, len(stock_tickers))
    best_returns = []
    elite_count = int(ELITISM_RATE * POPULATION_SIZE2)
    
    for gen in range(GENERATIONS):
        offspring = []
        offspring_mutation_rates = []
        
        # Generate offspring with self-adaptive mutation
        for i in range(OFFSPRING_SIZE):
            parent_idx = np.random.randint(0, POPULATION_SIZE2)
            child = mutate(population[parent_idx], mutation_rates[parent_idx])
            child_mutation_rate = self_adaptive_mutation(mutation_rates[parent_idx])
            offspring.append(child)
            offspring_mutation_rates.append(child_mutation_rate)
        
        # Combine parent population, offspring, and apply elitism
        combined_population = np.vstack([population, offspring])
        combined_mutation_rates = np.hstack([mutation_rates, offspring_mutation_rates])
        population_fitness = [fitness(ind, average_monthly_returns, cov_matrix) for ind in combined_population]
        
        elite_indices = np.argsort(population_fitness)[-elite_count:]
        selected_indices = np.random.choice(
            range(len(combined_population)), size=POPULATION_SIZE2 - elite_count, p=np.exp(population_fitness) / np.sum(np.exp(population_fitness))
        )
        
        new_population = np.vstack([combined_population[elite_indices], combined_population[selected_indices]])
        new_mutation_rates = np.hstack([combined_mutation_rates[elite_indices], combined_mutation_rates[selected_indices]])
        
        # Update population and mutation rates
        population = new_population
        mutation_rates = new_mutation_rates
        best_returns.append(population_fitness[elite_indices[-1]])
    
    return best_returns, population[np.argmax(population_fitness)]

# Measure expected return and convergence for multiple runs
all_best_returns = []
for run in range(NUM_RUNS):
    best_returns, best_weights = run_advanced_ep()
    all_best_returns.append(best_returns)

# Plot expected return of the optimized portfolio
plt.figure(figsize=(10, 6))
plt.plot(best_returns, label="Expected Return")
plt.title('Expected Return of Optimized Portfolio')
plt.xlabel('Generation')
plt.ylabel('Expected Return')
plt.legend()
plt.savefig(os.path.join(OUTPUT_FOLDER, "expected_return.png"))
plt.close()

# Plot convergence across multiple runs
for run, best_returns in enumerate(all_best_returns):
    plt.plot(best_returns, label=f"Run {run+1}")
plt.title('Convergence to Near-Optimal Solution')
plt.xlabel('Generation')
plt.ylabel('Expected Return')
plt.legend()
plt.savefig(os.path.join(OUTPUT_FOLDER, "convergence.png"))
plt.close()

# Plot consistency across runs
consistency = np.mean(all_best_returns, axis=0)
plt.figure(figsize=(10, 6))
plt.plot(consistency, label="Mean Expected Return Across Runs")
plt.fill_between(range(GENERATIONS), 
                 consistency - np.std(all_best_returns, axis=0), 
                 consistency + np.std(all_best_returns, axis=0), 
                 alpha=0.2, color="gray", label="Std. Dev.")
plt.title('Consistency of Results Across Runs')
plt.xlabel('Generation')
plt.ylabel('Expected Return')
plt.legend()
plt.savefig(os.path.join(OUTPUT_FOLDER, "consistency.png"))
plt.close()

# Create CSV file for results
results_df = pd.DataFrame({
    'Generation': range(GENERATIONS),
    'Mean Expected Return': consistency
})
results_df.to_csv(os.path.join(OUTPUT_FOLDER, "advanced_ep_optimization_results.csv"), index=False)
#---------------------
#---------------------
#---------------------
#---------------------
#---------------------3
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

# Set parameters for Basic Evolutionary Strategies (ES)
POPULATION_SIZE3 = 100          # Define population size
GENERATIONS = 200             # Define number of generations
SIGMA = 0.1                    # Mutation standard deviation
NUM_RUNS = 10                  # Number of runs to assess consistency
OUTPUT_FOLDER = "es_results_v3"
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# Define stock tickers and fetch data
stock_tickers = [
    'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 
    'META', 'NFLX', 'NVDA', 'BRK-B', 'JNJ', 
    'JPM', 'V', 'PG', 'DIS', 'ADBE', 
    'PYPL', 'INTC', 'CSCO', 'XOM', 'WMT'
]
stock_data = yf.download(stock_tickers, start='2018-01-01', end='2022-12-31')
monthly_returns = stock_data['Close'].pct_change().resample('M').sum()
cov_matrix = monthly_returns.cov()
average_monthly_returns = monthly_returns.mean()

# Basic ES Functions
def initialize_population(size, num_assets):
    return np.random.dirichlet(np.ones(num_assets), size=size)

def fitness(weights, returns, cov_matrix):
    portfolio_return = np.dot(weights, returns)
    portfolio_variance = np.dot(weights.T, np.dot(cov_matrix, weights))
    return portfolio_return / np.sqrt(portfolio_variance)  # Sharpe ratio

def mutate(weights, sigma):
    mutated_weights = weights + np.random.normal(0, sigma, len(weights))
    mutated_weights = np.abs(mutated_weights)  # Ensure weights remain positive
    return mutated_weights / np.sum(mutated_weights)  # Normalize weights

# Run Basic Evolutionary Strategies
def run_basic_es():
    population = initialize_population(POPULATION_SIZE3, len(stock_tickers))
    best_returns = []

    for gen in range(GENERATIONS):
        offspring = [mutate(ind, SIGMA) for ind in population]
        combined_population = np.vstack([population, offspring])
        population_fitness = [fitness(ind, average_monthly_returns, cov_matrix) for ind in combined_population]
        
        # Select the top individuals to form the new population
        best_indices = np.argsort(population_fitness)[-POPULATION_SIZE3:]
        population = combined_population[best_indices]
        best_returns.append(population_fitness[best_indices[-1]])
    
    return best_returns, population[np.argmax(population_fitness)]

# Measure expected return and convergence for multiple runs
all_best_returns = []
for run in range(NUM_RUNS):
    best_returns, best_weights = run_basic_es()
    all_best_returns.append(best_returns)

# Plot expected return of the optimized portfolio
plt.figure(figsize=(10, 6))
plt.plot(best_returns, label="Expected Return")
plt.title('Expected Return of Optimized Portfolio')
plt.xlabel('Generation')
plt.ylabel('Expected Return')
plt.legend()
plt.savefig(os.path.join(OUTPUT_FOLDER, "expected_return.png"))
plt.close()

# Plot convergence across multiple runs
for run, best_returns in enumerate(all_best_returns):
    plt.plot(best_returns, label=f"Run {run+1}")
plt.title('Convergence to Near-Optimal Solution')
plt.xlabel('Generation')
plt.ylabel('Expected Return')
plt.legend()
plt.savefig(os.path.join(OUTPUT_FOLDER, "convergence.png"))
plt.close()

# Plot consistency across runs
consistency = np.mean(all_best_returns, axis=0)
plt.figure(figsize=(10, 6))
plt.plot(consistency, label="Mean Expected Return Across Runs")
plt.fill_between(range(GENERATIONS), 
                 consistency - np.std(all_best_returns, axis=0), 
                 consistency + np.std(all_best_returns, axis=0), 
                 alpha=0.2, color="gray", label="Std. Dev.")
plt.title('Consistency of Results Across Runs')
plt.xlabel('Generation')
plt.ylabel('Expected Return')
plt.legend()
plt.savefig(os.path.join(OUTPUT_FOLDER, "consistency.png"))
plt.close()

# Create CSV file for results
results_df = pd.DataFrame({
    'Generation': range(GENERATIONS),
    'Mean Expected Return': consistency
})
results_df.to_csv(os.path.join(OUTPUT_FOLDER, "basic_es_optimization_results.csv"), index=False)
#---------------------
#---------------------
#---------------------
#---------------------
#---------------------4
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

# Set parameters for Advanced Evolutionary Strategies (ES)
POPULATION_SIZE4 = 250
GENERATIONS = 200
SIGMA_INITIAL = 0.1            # Initial mutation standard deviation
TAU = 1 / np.sqrt(2 * np.sqrt(POPULATION_SIZE4))  # Self-adaptive control parameter
NUM_RUNS = 10
OUTPUT_FOLDER = "advanced_es_results_v4"
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# Define stock tickers and fetch data
stock_tickers = [
    'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 
    'META', 'NFLX', 'NVDA', 'BRK-B', 'JNJ', 
    'JPM', 'V', 'PG', 'DIS', 'ADBE', 
    'PYPL', 'INTC', 'CSCO', 'XOM', 'WMT'
]
stock_data = yf.download(stock_tickers, start='2018-01-01', end='2022-12-31')
monthly_returns = stock_data['Close'].pct_change().resample('M').sum()
cov_matrix = monthly_returns.cov()
average_monthly_returns = monthly_returns.mean()

# Advanced ES Functions
def initialize_population(size, num_assets):
    weights = np.random.dirichlet(np.ones(num_assets), size=size)
    sigmas = np.full((size, num_assets), SIGMA_INITIAL)
    return weights, sigmas

def fitness(weights, returns, cov_matrix):
    portfolio_return = np.dot(weights, returns)
    portfolio_variance = np.dot(weights.T, np.dot(cov_matrix, weights))
    return portfolio_return / np.sqrt(portfolio_variance)  # Sharpe ratio

def mutate(weights, sigmas, tau):
    sigmas *= np.exp(tau * np.random.randn(len(weights)))
    mutated_weights = weights + sigmas * np.random.randn(len(weights))
    mutated_weights = np.abs(mutated_weights)  # Ensure weights are positive
    return mutated_weights / np.sum(mutated_weights), sigmas  # Normalize weights

def recombine(parents):
    return np.mean(parents, axis=0)

# Run Advanced Evolutionary Strategies
def run_advanced_es():
    weights, sigmas = initialize_population(POPULATION_SIZE4, len(stock_tickers))
    best_returns = []

    for gen in range(GENERATIONS):
        offspring_weights = []
        offspring_sigmas = []
        
        for i in range(POPULATION_SIZE4):
            # Select two random parents and recombine
            parents = np.random.choice(range(POPULATION_SIZE4), size=2, replace=False)
            child_weights = recombine([weights[parents[0]], weights[parents[1]]])
            child_sigmas = recombine([sigmas[parents[0]], sigmas[parents[1]]])
            
            # Mutate child
            child_weights, child_sigmas = mutate(child_weights, child_sigmas, TAU)
            offspring_weights.append(child_weights)
            offspring_sigmas.append(child_sigmas)
        
        combined_weights = np.vstack([weights, offspring_weights])
        combined_sigmas = np.vstack([sigmas, offspring_sigmas])
        population_fitness = [fitness(ind, average_monthly_returns, cov_matrix) for ind in combined_weights]
        
        # Select the top-performing individuals
        best_indices = np.argsort(population_fitness)[-POPULATION_SIZE4:]
        weights = combined_weights[best_indices]
        sigmas = combined_sigmas[best_indices]
        best_returns.append(population_fitness[best_indices[-1]])
    
    return best_returns, weights[np.argmax(population_fitness)]

# Measure expected return and convergence for multiple runs
all_best_returns = []
for run in range(NUM_RUNS):
    best_returns, best_weights = run_advanced_es()
    all_best_returns.append(best_returns)

# Plot expected return of the optimized portfolio
plt.figure(figsize=(10, 6))
plt.plot(best_returns, label="Expected Return")
plt.title('Expected Return of Optimized Portfolio')
plt.xlabel('Generation')
plt.ylabel('Expected Return')
plt.legend()
plt.savefig(os.path.join(OUTPUT_FOLDER, "expected_return.png"))
plt.close()

# Plot convergence across multiple runs
for run, best_returns in enumerate(all_best_returns):
    plt.plot(best_returns, label=f"Run {run+1}")
plt.title('Convergence to Near-Optimal Solution')
plt.xlabel('Generation')
plt.ylabel('Expected Return')
plt.legend()
plt.savefig(os.path.join(OUTPUT_FOLDER, "convergence.png"))
plt.close()

# Plot consistency across runs
consistency = np.mean(all_best_returns, axis=0)
plt.figure(figsize=(10, 6))
plt.plot(consistency, label="Mean Expected Return Across Runs")
plt.fill_between(range(GENERATIONS), 
                 consistency - np.std(all_best_returns, axis=0), 
                 consistency + np.std(all_best_returns, axis=0), 
                 alpha=0.2, color="gray", label="Std. Dev.")
plt.title('Consistency of Results Across Runs')
plt.xlabel('Generation')
plt.ylabel('Expected Return')
plt.legend()
plt.savefig(os.path.join(OUTPUT_FOLDER, "consistency.png"))
plt.close()

# Create CSV file for results
results_df = pd.DataFrame({
    'Generation': range(GENERATIONS),
    'Mean Expected Return': consistency
})
results_df.to_csv(os.path.join(OUTPUT_FOLDER, "advanced_es_optimization_results.csv"), index=False)
#---------------------
#---------------------
#---------------------
#---------------------
#---------------------5
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

# Set parameters for (μ + λ) Evolutionary Strategies (ES)
POPULATION_SIZE5 = 50  # μ
OFFSPRING_SIZE = 100  # λ
GENERATIONS = 200
SIGMA_INITIAL = 0.1  # Initial mutation standard deviation
NUM_RUNS = 10
OUTPUT_FOLDER = "mu_lambda_es_results_v5"
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# Define stock tickers and fetch data
stock_tickers = [
    'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 
    'META', 'NFLX', 'NVDA', 'BRK-B', 'JNJ', 
    'JPM', 'V', 'PG', 'DIS', 'ADBE', 
    'PYPL', 'INTC', 'CSCO', 'XOM', 'WMT'
]
stock_data = yf.download(stock_tickers, start='2018-01-01', end='2022-12-31')
monthly_returns2 = stock_data['Close'].pct_change().resample('ME').sum()
cov_matrix = monthly_returns2.cov()
average_monthly_returns2 = monthly_returns2.mean()

# (μ + λ) ES Functions
def initialize_population(size, num_assets):
    weights = np.random.dirichlet(np.ones(num_assets), size=size)
    sigmas = np.full((size, num_assets), SIGMA_INITIAL)
    return weights, sigmas

def fitness(weights, returns, cov_matrix):
    portfolio_return = np.dot(weights, returns)
    portfolio_variance = np.dot(weights.T, np.dot(cov_matrix, weights))
    return portfolio_return / np.sqrt(portfolio_variance)  # Sharpe ratio

def mutate(weights, sigma):
    mutated_weights = weights + sigma * np.random.randn(len(weights))
    mutated_weights = np.abs(mutated_weights)  # Ensure weights are positive
    mutated_weights /= np.sum(mutated_weights)  # Normalize weights
    return mutated_weights, sigma  # Return both weights and sigma


# Run (μ + λ) Evolutionary Strategies
def run_mu_lambda_es():
    weights, sigmas = initialize_population(POPULATION_SIZE5, len(stock_tickers))
    best_returns = []

    for gen in range(GENERATIONS):
        offspring_weights = []
        offspring_sigmas = []
        
        for i in range(OFFSPRING_SIZE):
            parent_idx = np.random.choice(range(POPULATION_SIZE5))
            offspring_weight, offspring_sigma = mutate(weights[parent_idx], sigmas[parent_idx])
            offspring_weights.append(offspring_weight)
            offspring_sigmas.append(offspring_sigma)
        
        combined_weights = np.vstack([weights, offspring_weights])
        population_fitness = [fitness(ind, average_monthly_returns2, cov_matrix) for ind in combined_weights]
        
        # Select the top-performing individuals from parents and offspring
        best_indices = np.argsort(population_fitness)[-POPULATION_SIZE5:]
        weights = combined_weights[best_indices]
        best_returns.append(population_fitness[best_indices[-1]])
    
    return best_returns, weights[np.argmax(population_fitness)]

# Measure expected return and convergence for multiple runs
all_best_returns = []
for run in range(NUM_RUNS):
    best_returns, best_weights = run_mu_lambda_es()
    all_best_returns.append(best_returns)

# Plot expected return of the optimized portfolio
plt.figure(figsize=(10, 6))
plt.plot(best_returns, label="Expected Return")
plt.title('Expected Return of Optimized Portfolio')
plt.xlabel('Generation')
plt.ylabel('Expected Return')
plt.legend()
plt.savefig(os.path.join(OUTPUT_FOLDER, "expected_return.png"))
plt.close()

# Plot convergence across multiple runs
for run, best_returns in enumerate(all_best_returns):
    plt.plot(best_returns, label=f"Run {run+1}")
plt.title('Convergence to Near-Optimal Solution')
plt.xlabel('Generation')
plt.ylabel('Expected Return')
plt.legend()
plt.savefig(os.path.join(OUTPUT_FOLDER, "convergence.png"))
plt.close()

# Plot consistency across runs
consistency = np.mean(all_best_returns, axis=0)
plt.figure(figsize=(10, 6))
plt.plot(consistency, label="Mean Expected Return Across Runs")
plt.fill_between(range(GENERATIONS), 
                 consistency - np.std(all_best_returns, axis=0), 
                 consistency + np.std(all_best_returns, axis=0), 
                 alpha=0.2, color="gray", label="Std. Dev.")
plt.title('Consistency of Results Across Runs')
plt.xlabel('Generation')
plt.ylabel('Expected Return')
plt.legend()
plt.savefig(os.path.join(OUTPUT_FOLDER, "consistency.png"))
plt.close()

# Create CSV file for results
results_df = pd.DataFrame({
    'Generation': range(GENERATIONS),
    'Mean Expected Return': consistency
})
results_df.to_csv(os.path.join(OUTPUT_FOLDER, "mu_lambda_es_optimization_results.csv"), index=False)
#---------------------
#---------------------
#---------------------
#---------------------
#---------------------6
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

# Set parameters for (μ, λ) Evolutionary Strategies (ES)
POPULATION_SIZE6 = 50  # μ
OFFSPRING_SIZE = 100  # λ
GENERATIONS = 200
SIGMA_INITIAL = 0.1  # Initial mutation standard deviation
NUM_RUNS = 10
OUTPUT_FOLDER = "mu_comma_lambda_es_results_v6"
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# Define stock tickers and fetch data
stock_tickers = [
    'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 
    'META', 'NFLX', 'NVDA', 'BRK-B', 'JNJ', 
    'JPM', 'V', 'PG', 'DIS', 'ADBE', 
    'PYPL', 'INTC', 'CSCO', 'XOM', 'WMT'
]
stock_data = yf.download(stock_tickers, start='2018-01-01', end='2022-12-31')
monthly_returns2 = stock_data['Close'].pct_change().resample('ME').sum()
cov_matrix = monthly_returns2.cov()
average_monthly_returns2 = monthly_returns2.mean()

# (μ, λ) ES Functions
def initialize_population(size, num_assets):
    weights = np.random.dirichlet(np.ones(num_assets), size=size)
    sigmas = np.full((size, num_assets), SIGMA_INITIAL)
    return weights, sigmas

def fitness(weights, returns, cov_matrix):
    portfolio_return = np.dot(weights, returns)
    portfolio_variance = np.dot(weights.T, np.dot(cov_matrix, weights))
    return portfolio_return / np.sqrt(portfolio_variance)  # Sharpe ratio

def mutate(weights, sigma):
    mutated_weights = weights + sigma * np.random.randn(len(weights))
    mutated_weights = np.abs(mutated_weights)  # Ensure weights are positive
    mutated_weights /= np.sum(mutated_weights)  # Normalize weights
    return mutated_weights, sigma  # Return both weights and sigma


# Run (μ, λ) Evolutionary Strategies
def run_mu_comma_lambda_es():
    weights, sigmas = initialize_population(POPULATION_SIZE6, len(stock_tickers))
    best_returns = []

    for gen in range(GENERATIONS):
        offspring_weights = []
        offspring_sigmas = []
        
        for i in range(OFFSPRING_SIZE):
            parent_idx = np.random.choice(range(POPULATION_SIZE6))
            offspring_weight, offspring_sigma = mutate(weights[parent_idx], sigmas[parent_idx])
            offspring_weights.append(offspring_weight)
            offspring_sigmas.append(offspring_sigma)
        
        offspring_fitness = [fitness(ind, average_monthly_returns2, cov_matrix) for ind in offspring_weights]
        
        # Select the best μ offspring
        best_indices = np.argsort(offspring_fitness)[-POPULATION_SIZE6:]
        weights = np.array(offspring_weights)[best_indices]
        best_returns.append(offspring_fitness[best_indices[-1]])
    
    return best_returns, offspring_weights[np.argmax(offspring_fitness)]

# Measure expected return and convergence for multiple runs
all_best_returns = []
for run in range(NUM_RUNS):
    best_returns, best_weights = run_mu_comma_lambda_es()
    all_best_returns.append(best_returns)

# Plot expected return of the optimized portfolio
plt.figure(figsize=(10, 6))
plt.plot(best_returns, label="Expected Return")
plt.title('Expected Return of Optimized Portfolio')
plt.xlabel('Generation')
plt.ylabel('Expected Return')
plt.legend()
plt.savefig(os.path.join(OUTPUT_FOLDER, "expected_return.png"))
plt.close()

# Plot convergence across multiple runs
for run, best_returns in enumerate(all_best_returns):
    plt.plot(best_returns, label=f"Run {run+1}")
plt.title('Convergence to Near-Optimal Solution')
plt.xlabel('Generation')
plt.ylabel('Expected Return')
plt.legend()
plt.savefig(os.path.join(OUTPUT_FOLDER, "convergence.png"))
plt.close()

# Plot consistency across runs
consistency = np.mean(all_best_returns, axis=0)
plt.figure(figsize=(10, 6))
plt.plot(consistency, label="Mean Expected Return Across Runs")
plt.fill_between(range(GENERATIONS), 
                 consistency - np.std(all_best_returns, axis=0), 
                 consistency + np.std(all_best_returns, axis=0), 
                 alpha=0.2, color="gray", label="Std. Dev.")
plt.title('Consistency of Results Across Runs')
plt.xlabel('Generation')
plt.ylabel('Expected Return')
plt.legend()
plt.savefig(os.path.join(OUTPUT_FOLDER, "consistency.png"))
plt.close()

# Create CSV file for results
results_df = pd.DataFrame({
    'Generation': range(GENERATIONS),
    'Mean Expected Return': consistency
})
results_df.to_csv(os.path.join(OUTPUT_FOLDER, "mu_comma_lambda_es_optimization_results.csv"), index=False)
