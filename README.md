# MULTIC-JET-CUDA

MULTIC-JET-CUDA é um projeto para simulações de jatos multicomponentes usando o método Lattice Boltzmann (LBM), implementado com suporte para GPUs, permitindo a execução eficiente de simulações computacionalmente intensivas. O projeto tem suporte D3Q19 para ambos os componentes.

## Estrutura do Projeto

- **src/**: Contém o código-fonte principal, incluindo kernels CUDA e scripts de compilação.
- **bin/**: Diretório de saída para os binários compilados e resultados de simulação.
- **post/**: Scripts para pós-processamento dos resultados da simulação.
- **pipeline.sh**: Script principal para executar o pipeline de compilação, simulação e pós-processamento.

## Como Executar

1. **Compilar e Executar a Simulação**:

   Use o script `pipeline.sh` para compilar e executar o simulador:

   ```bash
   ./pipeline.sh <velocity_set> <id>
   ```

   - **`<velocity_set>`**: *Velocity set* relacionado aos componentes.
   - **`<id>`**: Identificador único para a simulação (exemplo: `000`).

   Exemplo:

   ```bash
   ./pipeline.sh D3Q19 000
   ```

2. **Resultados**:
   - Saídas de simulação serão salvas em:
     - `bin/<velocity_set>/<id>/`

3. **Pós-Processamento**:
   O script também executará automaticamente o pós-processamento.

## Exemplos de Execução

- **Executar uma simulação**:
  ```bash
  ./pipeline.sh D3Q19 000
  ```


