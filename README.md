# Bubble-GPU

Bubble-GPU é um projeto para simulações de fluidos usando o método Lattice Boltzmann (LBM), implementado com suporte para GPUs, permitindo a execução eficiente de simulações computacionalmente intensivas. O foco atual está na simulação de bolhas estacionárias e oscilantes em 3D utilizando os modelos D3Q19 e D3Q27.

## Estrutura do Projeto

- **src/**: Contém o código-fonte principal, incluindo kernels CUDA e scripts de compilação.
- **bin/**: Diretório de saída para os binários compilados e resultados de simulação.
- **post/**: Scripts para pós-processamento dos resultados da simulação.
- **pipeline.sh**: Script principal para executar o pipeline de compilação, simulação e pós-processamento.

## Como Executar

1. **Compilar e Executar a Simulação**:

   Use o script `pipeline.sh` para compilar e executar o simulador:

   ```bash
   ./pipeline.sh <fluid_model> <phase_model> <id>
   ```

   - **`<fluid_model>`**: Modelo de fluido (exemplo: `FD3Q19`).
   - **`<phase_model>`**: Modelo de fase (exemplo: `PD3Q15`).
   - **`<id>`**: Identificador único para a simulação (exemplo: `000`).

   Exemplo:

   ```bash
   ./pipeline.sh FD3Q19 PD3Q15 001
   ```

2. **Resultados**:
   - Saídas de simulação serão salvas em:
     - `bin/<fluid_model>_<phase_model>/<id>/`

3. **Pós-Processamento**:
   O script também executará automaticamente o pós-processamento.

## Exemplos de Execução

- **Executar uma simulação**:
  ```bash
  ./pipeline.sh FD3Q19 PD3Q15 001
  ```


