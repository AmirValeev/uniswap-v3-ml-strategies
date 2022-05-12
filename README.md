## Uniswap V3 strategy framework

This repository contains several python scripts to simulate the performance of Uniswap v3 liquidity provision strategies performance and evaluate risks. The main scripts of the package are:

1. [ActiveStrategyFramework.py](ActiveStrategyFramework.py) base code of the framework which executues a strategy, conducting either back-testing simulations (```simulate_strategy``` function and passing in historical swap data), or conducting a live implementation of the strategy.
2. [ML_Strategy.py](ML_Strategy.py) strategy uses an AR(1)-GARCH(1,1) model to predict volatility alongside with XGBoost to predict base asset price movement.
3. [GetPoolData.py](GetPoolData.py) which downloads the data necessary for the simulations. 
4. [UNI_v3_funcs.py](UNI_v3_funcs.py) python implementation of Uniswap v3's [liquidity math](https://github.com/Uniswap/uniswap-v3-periphery/blob/main/contracts/libraries/LiquidityAmounts.sol). 

In order to provide an illustration of potential usage, i have included Jupyter Notebook that show how to use the framework:
- [4_ML_Strategy.ipynb](4_ML_Strategy_Example.ipynb) 

Have constructed a flexible framework for active LP strategy simulations that uses **the full Uniswap v3 swap history** in order to improve accurracy of fee income. Thefore simulations are available in the time period since Unsiwap v3 was released (May 5th 2021 is when swap data starts to show up consistently). 
## Live running with Mlflow and Airflow

For live performance i choosed pattern with:

- Every week XGBoost model retrained on new coming data
- Every 10 min we predict and reset strategy

To see instructions see `airflow_mlflow.md` 

## Data & simulating a different pool


**The Graph + Bitquery + Flipside Crypto**

The pattern to use these data sources can be seen in [2_AutoRegressive_Strategy_Example.ipynb](2_AutoRegressive_Strategy_Example.ipynb). The data sources are:

- **[The Graph](https://thegraph.com/legacy-explorer/subgraph/uniswap/uniswap-v3):** We obtain the full history of Uniswap v3 swaps from whatever pool we need, in order to accurately simulate the performance of the simulated strategy.
- **[Bitquery](https://graphql.bitquery.io/ide):** We obtain historical token prices from Uniswap v2 and v3. 
- **[Flipside Crypto](https://app.flipsidecrypto.com/velocity):** We obtain the virtual liquidity of the pool at every block, which is used to approximate the fee income earned in the pool, as described in their [documentation](https://docs.flipsidecrypto.com/our-data/tables/uniswap-v3-tables/pool-stats).

*Instructions*
1. Obtain a free API key from [Bitquery](https://graphql.bitquery.io/ide).
2. Save it in a file in ```config.py``` in the directory where the ActiveStrategyFramework is stored as a variable called ```BITQUERY_API_TOKEN``` (eg. ```BITQUERY_API_TOKEN = XXXXXXXX```).
3. Generate a new Flipside Crypto query, with the ```pool_address``` for the pair that you are interested. Note that due to a 100,000 row limit, we generate two queries for the USDC/WETH 0.3%, which explains the ```BLOCK_ID``` condition, to split the data into reasonable chunks. A less active pool might not need this split.

