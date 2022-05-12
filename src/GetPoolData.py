import pandas as pd
from datetime import datetime, timedelta
import requests
import pickle
import importlib
from itertools import compress
import time
import os
import math

##############################################################
# Pull Uniswap v3 pool data from Google Bigquery
# Have options for Ethereum Mainnet and Polygon
##############################################################


def download_bigquery_price_mainnet(
    contract_address, date_begin, date_end, block_start
):
    """
    Internal function to query Google Bigquery for the swap history of a Uniswap v3 pool between two dates starting from a particular block from Ethereum Mainnet.
    Use GetPoolData.get_pool_data_bigquery which preprocesses the data in order to conduct simualtions with the Active Strategy Framework.
    """

    from google.cloud import bigquery

    client = bigquery.Client()

    query = (
        """
            SELECT *
            FROM blockchain-etl.ethereum_uniswap.UniswapV3Pool_event_Swap
            where contract_address = lower('"""
        + contract_address.lower()
        + """') and
              block_timestamp >= '"""
        + str(date_begin)
        + """' and block_timestamp <= '"""
        + str(date_end)
        + """' and block_number >= """
        + str(block_start)
        + """
            """
    )
    query_job = client.query(query)  # Make an API request.
    return query_job.to_dataframe(create_bqstorage_client=False)


def download_bigquery_price_polygon(
    contract_address, date_begin, date_end, block_start
):
    """
    Internal function to query Google Bigquery for the swap history of a Uniswap v3 pool between two dates starting from a particular block from Polygon.
    Use GetPoolData.get_pool_data_bigquery which preprocesses the data in order to conduct simualtions with the Active Strategy Framework.
    """

    from google.cloud import bigquery

    client = bigquery.Client()
    query = (
        '''SELECT
      block_number,
      transaction_index,
      log_index,
      block_hash,
      transaction_hash,
      address,
      block_timestamp,
      '0x' || RIGHT(topics[SAFE_OFFSET(1)],40) AS sender,
      '0x' || RIGHT(topics[SAFE_OFFSET(1)],40) AS recipient,
      '0x' || SUBSTR(DATA, 3, 64) AS amount0,
      '0x' || SUBSTR(DATA, 67, 64) AS amount1,
      '0x' || SUBSTR(DATA,131,64) AS sqrtPriceX96,
      '0x' || SUBSTR(DATA,195,64) AS liquidity,
      '0x' || SUBSTR(DATA,259,64) AS tick
    FROM
      public-data-finance.crypto_polygon.logs
    WHERE
      topics[SAFE_OFFSET(0)] = '0xc42079f94a6350d7e6235f29174924f928cc2ac818eb64fed8004e115fbcca67'
      AND DATE(block_timestamp) >=  DATE("'''
        + date_begin
        + '''")
      AND DATE(block_timestamp) <=  DATE("'''
        + date_end
        + """")
      AND block_number          >=  """
        + str(block_start)
        + '''
      AND address = "'''
        + contract_address
        + """"
     """
    )
    query_job = client.query(query)  # Make an API request.

    result = query_job.to_dataframe(create_bqstorage_client=False)
    result["amount0"] = result["amount0"].apply(signed_int)
    result["amount1"] = result["amount1"].apply(signed_int)
    result["sqrtPriceX96"] = result["sqrtPriceX96"].apply(signed_int)
    result["liquidity"] = result["liquidity"].apply(signed_int)
    result["tick"] = result["tick"].apply(signed_int)

    return result


def get_pool_data_bigquery(
    contract_address,
    date_begin,
    date_end,
    decimals_0,
    decimals_1,
    network="mainnet",
    block_start=0,
):

    """
    Queries Google Bigquery for the swap history of Uniswap v3 pool between two dates starting from a particular block from either Ethereum Mainnet or Polygon.
    Preprocesses data to have decimal adjusted amounts and liquidity values.
    """

    if network == "mainnet":
        resulting_data = download_bigquery_price_mainnet(
            contract_address.lower(), date_begin, date_end, block_start
        )
    elif network == "polygon":
        resulting_data = download_bigquery_price_polygon(
            contract_address.lower(), date_begin, date_end, block_start
        )
    else:
        raise ValueError("Unsupported Network:" + network)

    DECIMAL_ADJ = 10 ** (decimals_1 - decimals_0)
    resulting_data["sqrtPriceX96_float"] = resulting_data["sqrtPriceX96"].astype(float)
    resulting_data["quotePrice"] = (
        (resulting_data["sqrtPriceX96_float"] / 2**96) ** 2
    ) / DECIMAL_ADJ
    resulting_data["block_date"] = pd.to_datetime(resulting_data["block_timestamp"])
    resulting_data = resulting_data.set_index("block_date", drop=False).sort_index()

    resulting_data["tick_swap"] = resulting_data["tick"].astype(int)
    resulting_data["amount0"] = resulting_data["amount0"].astype(float)
    resulting_data["amount1"] = resulting_data["amount1"].astype(float)
    resulting_data["amount0_adj"] = (
        resulting_data["amount0"].astype(float) / 10**decimals_0
    )
    resulting_data["amount1_adj"] = (
        resulting_data["amount1"].astype(float) / 10**decimals_1
    )
    resulting_data["virtual_liquidity"] = resulting_data["liquidity"].astype(float)
    resulting_data["virtual_liquidity_adj"] = resulting_data["liquidity"].astype(
        float
    ) / (10 ** ((decimals_0 + decimals_1) / 2))
    resulting_data["token_in"] = resulting_data.apply(
        lambda x: "token0" if (x["amount0_adj"] < 0) else "token1", axis=1
    )
    resulting_data["traded_in"] = resulting_data.apply(
        lambda x: -x["amount0_adj"] if (x["amount0_adj"] < 0) else -x["amount1_adj"],
        axis=1,
    ).astype(float)

    return resulting_data


def signed_int(h):
    """
    Converts hex values to signed integers.
    """
    s = bytes.fromhex(h[2:])
    i = int.from_bytes(s, "big", signed=True)
    return i


##############################################################
# Get Swaps from Uniswap v3's subgraph, and liquidity at each swap from Flipside Crypto
##############################################################


def query_univ3_graph(query: str, variables=None, network="mainnet") -> dict:
    """
    Internal function to query The Graph's Uniswap v3 subgraph on either mainnet or arbitrum.
    Use GetPoolData.get_pool_data_flipside which preprocesses the data in order to conduct simualtions with the Active Strategy Framework.
    """

    if network == "mainnet":
        univ3_graph_url = "https://api.thegraph.com/subgraphs/name/uniswap/uniswap-v3"
    elif network == "arbitrum":
        univ3_graph_url = (
            "https://api.thegraph.com/subgraphs/name/ianlapham/uniswap-arbitrum-one"
        )

    if variables:
        params = {"query": query, "variables": variables}
    else:
        params = {"query": query}

    response = requests.post(univ3_graph_url, json=params)
    return response.json()


def get_swap_data(contract_address, file_name, DOWNLOAD_DATA=True, network="mainnet"):
    """
    Internal function to query full history of swap data from Uniswap v3's subgraph.
    Use GetPoolData.get_pool_data_flipside which preprocesses the data in order to conduct simualtions with the Active Strategy Framework.
    """

    request_swap = []

    if DOWNLOAD_DATA:

        current_payload = generate_first_event_payload("swaps", contract_address)
        current_id = query_univ3_graph(current_payload, network=network)["data"][
            "pool"
        ]["swaps"][0]["id"]
        finished = False

        while not finished:
            current_payload = generate_event_payload(
                "swaps", contract_address, str(1000)
            )
            response = query_univ3_graph(
                current_payload, variables={"paginateId": current_id}, network=network
            )["data"]["pool"]["swaps"]

            if len(response) == 0:
                finished = True
            else:
                current_id = response[-1]["id"]
                request_swap.extend(response)

            with open("./data/" + file_name + "_swap.pkl", "wb") as output:
                pickle.dump(request_swap, output, pickle.HIGHEST_PROTOCOL)
    else:
        with open("./data/" + file_name + "_swap.pkl", "rb") as input:
            request_swap = pickle.load(input)

    return pd.DataFrame(request_swap)


def get_liquidity_flipside(flipside_query, file_name, DOWNLOAD_DATA=True):
    """
    Internal function to query full history of liquidity values from Flipside Crypto's Uniswap v3's databases.
    Use GetPoolData.get_pool_data_flipside which preprocesses the data in order to conduct simualtions with the Active Strategy Framework.
    """

    if DOWNLOAD_DATA:
        request_stats = [pd.DataFrame(requests.get(x).json()) for x in flipside_query]
        with open("./data/" + file_name + "_liquidity.pkl", "wb") as output:
            pickle.dump(request_stats, output, pickle.HIGHEST_PROTOCOL)
    else:
        with open("./data/" + file_name + "_liquidity.pkl", "rb") as input:
            request_stats = pickle.load(input)

    stats_data = pd.concat(request_stats)

    return stats_data


def get_pool_data_flipside(
    contract_address, flipside_query, file_name, DOWNLOAD_DATA=True
):
    """
    Queries Uniswap v3's subgraph for swap data and Flipside Crypto's queries to find liquidity in order to conduct simulations using the Active Strategy Framework.
    """

    # Download  events
    swap_data = get_swap_data(contract_address, file_name, DOWNLOAD_DATA)
    swap_data["time_pd"] = pd.to_datetime(
        swap_data["timestamp"], unit="s", origin="unix", utc=True
    )
    swap_data = swap_data.set_index("time_pd")
    swap_data["tick_swap"] = swap_data["tick"]
    swap_data = swap_data.sort_index()

    # Download pool liquidity data
    stats_data = get_liquidity_flipside(flipside_query, file_name, DOWNLOAD_DATA)
    stats_data["time_pd"] = pd.to_datetime(
        stats_data["BLOCK_TIMESTAMP"], origin="unix", utc=True
    )
    stats_data = stats_data.set_index("time_pd")
    stats_data = stats_data.sort_index()
    stats_data["tick_pool"] = stats_data["TICK"]

    full_data = pd.merge_asof(
        swap_data,
        stats_data[["VIRTUAL_LIQUIDITY_ADJUSTED", "tick_pool"]],
        on="time_pd",
        direction="backward",
        allow_exact_matches=False,
    )
    full_data = full_data.set_index("time_pd")
    # token with negative amounts is the token being swapped in
    full_data["tick_swap"] = full_data["tick_swap"].astype(int)
    full_data["amount0"] = full_data["amount0"].astype(float)
    full_data["amount1"] = full_data["amount1"].astype(float)
    full_data["token_in"] = full_data.apply(
        lambda x: "token0" if (x["amount0"] < 0) else "token1", axis=1
    )

    return full_data


def generate_event_payload(event, address, n_query):
    payload = (
        '''
            query($paginateId: String!){
              pool(id:"'''
        + address
        + """"){
                """
        + event
        + """(
                  first: """
        + n_query
        + """
                  orderBy: id
                  orderDirection: asc
                  where: {
                    id_gt: $paginateId
                  }
                ) {
                  id
                  timestamp
                  tick
                  amount0
                  amount1
                  amountUSD
                }
              }
            }"""
    )
    return payload


def generate_first_event_payload(event, address):
    payload = (
        '''query{
                      pool(id:"'''
        + address
        + """"){
                      """
        + event
        + """(
                      first: 1
                      orderBy: id
                      orderDirection: asc
                        ) {
                          id
                          timestamp
                          tick
                          amount0
                          amount1
                          amountUSD
                        }
                      }
                    }"""
    )
    return payload


##########################
# Uniswap v2
##########################


def query_univ2_graph(query: str, variables=None) -> dict:
    """
    Internal function to query The Graph's Uniswap v2 subgraph on mainnet.
    Use GetPoolData.get_swap_data_univ2 which preprocesses the data in order to conduct simualtions with the Active Strategy Framework.
    """

    univ2_graph_url = "https://api.thegraph.com/subgraphs/name/uniswap/uniswap-v2"

    if variables:
        params = {"query": query, "variables": variables}
    else:
        params = {"query": query}

    response = requests.post(univ2_graph_url, json=params)

    return response.json()


def download_swap_univ2_subgraph(
    contract_address, file_name, date_begin, date_end, DOWNLOAD_DATA=True
):
    """
    Internal function to query the history of swap data from Uniswap v2's subgraph between begin_date and end_date.
    Use GetPoolData.get_swap_data_univ2 which preprocesses the data in order to conduct simualtions with the Active Strategy Framework.
    """

    request_swap = []

    if DOWNLOAD_DATA:

        current_payload = generate_first_swap_univ2_payload(
            contract_address, date_begin, date_end
        )
        current_id = query_univ2_graph(current_payload)["data"]["swaps"][0]["id"]
        finished = False

        while not finished:
            current_payload = generate_swap_univ2_payload(
                contract_address, date_begin, date_end, str(1000)
            )
            response = query_univ2_graph(
                current_payload, variables={"paginateId": current_id}
            )["data"]["swaps"]

            if len(response) == 0:
                finished = True
            else:
                current_id = response[-1]["id"]
                request_swap.extend(response)

            with open("./data/" + file_name + "_swap_v2.pkl", "wb") as output:
                pickle.dump(request_swap, output, pickle.HIGHEST_PROTOCOL)
    else:
        with open("./data/" + file_name + "_swap_v2.pkl", "rb") as input:
            request_swap = pickle.load(input)

    return pd.DataFrame(request_swap)


def get_swap_data_univ2(
    contract_address, file_name, date_begin, date_end, DOWNLOAD_DATA=True
):
    """
    Queries Uniswap v2's subgraph for swap data in order to conduct simulations using the Active Strategy Framework.
    """

    swap_data = download_swap_univ2_subgraph(
        contract_address, file_name, date_begin, date_end, DOWNLOAD_DATA
    )
    swap_data["time_pd"] = pd.to_datetime(
        swap_data["timestamp"], unit="s", origin="unix", utc=True
    )
    swap_data = swap_data.set_index("time_pd", drop=False)
    swap_data = swap_data.sort_index()

    swap_data["token_in"] = swap_data.apply(
        lambda x: "token0" if float(x["amount0In"]) > 0 else "token1", axis=1
    )
    swap_data["amount0"] = swap_data.apply(
        lambda x: -float(x["amount0In"])
        if x["token_in"] == "token0"
        else float(x["amount0Out"]),
        axis=1,
    )
    swap_data["amount1"] = swap_data.apply(
        lambda x: float(x["amount1Out"])
        if x["token_in"] == "token0"
        else -float(x["amount1In"]),
        axis=1,
    )
    swap_data["traded_in"] = swap_data.apply(
        lambda x: -x["amount0"] if (x["amount0"] < 0) else -x["amount1"], axis=1
    ).astype(float)

    return swap_data


def generate_swap_univ2_payload(address, date_begin, date_end, n_query):
    """
    Internal function that generates GraphQL queries to to query The Graph's Uniswap v2 subgraph on mainnet.
    """

    date_begin_fmt = str(int(pd.Timestamp(date_begin).timestamp()))
    date_end_fmt = str(int(pd.Timestamp(date_end).timestamp()))

    payload = (
        """
        query($paginateId: String!){                   
          swaps(
          first: """
        + n_query
        + '''
          orderBy: id
          orderDirection: asc
          where:{
              pair:"'''
        + address
        + '''", 
              id_gt: $paginateId,
              timestamp_gte:"'''
        + date_begin_fmt
        + '''",
              timestamp_lte:"'''
        + date_end_fmt
        + """"
              }
            ) {
              id
              timestamp
              amount0In
              amount1In
              amount0Out
              amount1Out
              amountUSD
            }
          }"""
    )

    return payload


def generate_first_swap_univ2_payload(address, date_begin, date_end):
    """
    Internal function that generates GraphQL queries to to query The Graph's Uniswap v2 subgraph on mainnet.
    """

    date_begin_fmt = str(int(pd.Timestamp(date_begin).timestamp()))
    date_end_fmt = str(int(pd.Timestamp(date_end).timestamp()))

    payload = (
        '''query{                   
                      swaps(
                      first: 1
                      orderBy: id
                      orderDirection: asc
                      where:{pair:"'''
        + address
        + """",
                             timestamp_gte:"""
        + date_begin_fmt
        + """,
                             timestamp_lte:"""
        + date_end_fmt
        + """}
                        ) {
                          id
                          timestamp
                          amount0In
                          amount1In
                          amount0Out
                          amount1Out
                          amountUSD
                        }
                      }"""
    )

    return payload


##############################################################
# Get Price Data from Bitquery
##############################################################
def get_price_data_bitquery(
    token_0_address,
    token_1_address,
    date_begin,
    date_end,
    api_token,
    file_name,
    DOWNLOAD_DATA=True,
    RATE_LIMIT=False,
    exchange_to_query="Uniswap",
):
    """
    Queries the price history of a pair of ERC20's (located at token_0_address and token_1_address) in exchange_to_query (defaults to all Uniswap versions on mainnet) between begin_date and end_date on Bitquery.
    """
    request = []
    max_rows_bitquery = 10000

    if DOWNLOAD_DATA:
        # Paginate using limit and an offset
        offset = 0
        current_request = run_bitquery_query(
            generate_price_payload(
                token_0_address,
                token_1_address,
                date_begin,
                date_end,
                offset,
                exchange_to_query,
            ),
            api_token,
        )
        request.append(current_request)

        # When a request has less than 10,000 rows we are at the last one
        while (
            len(current_request["data"]["ethereum"]["dexTrades"]) == max_rows_bitquery
        ):
            current_request = run_bitquery_query(
                generate_price_payload(
                    token_0_address,
                    token_1_address,
                    date_begin,
                    date_end,
                    offset,
                    exchange_to_query,
                ),
                api_token,
            )
            request.append(current_request)
            offset += max_rows_bitquery
            if RATE_LIMIT:
                time.sleep(5)

        with open("./data/" + file_name + "_1min.pkl", "wb") as output:
            pickle.dump(request, output, pickle.HIGHEST_PROTOCOL)

    else:
        with open("./data/" + file_name + "_1min.pkl", "rb") as input:
            request = pickle.load(input)

    # Prepare data for strategy:
    # Collect json data and add to a pandas Data Frame

    requests_with_data = [len(x["data"]["ethereum"]["dexTrades"]) > 0 for x in request]
    relevant_requests = list(compress(request, requests_with_data))

    price_data = pd.concat(
        [
            pd.DataFrame(
                {
                    "time": [
                        x["timeInterval"]["minute"]
                        for x in request_price["data"]["ethereum"]["dexTrades"]
                    ],
                    "baseCurrency": [
                        x["baseCurrency"]["symbol"]
                        for x in request_price["data"]["ethereum"]["dexTrades"]
                    ],
                    "quoteCurrency": [
                        x["quoteCurrency"]["symbol"]
                        for x in request_price["data"]["ethereum"]["dexTrades"]
                    ],
                    "quoteAmount": [
                        x["quoteAmount"]
                        for x in request_price["data"]["ethereum"]["dexTrades"]
                    ],
                    "baseAmount": [
                        x["baseAmount"]
                        for x in request_price["data"]["ethereum"]["dexTrades"]
                    ],
                    "tradeAmount": [
                        x["tradeAmount"]
                        for x in request_price["data"]["ethereum"]["dexTrades"]
                    ],
                    "quotePrice": [
                        x["quotePrice"]
                        for x in request_price["data"]["ethereum"]["dexTrades"]
                    ],
                }
            )
            for request_price in relevant_requests
        ]
    )

    price_data["time"] = pd.to_datetime(price_data["time"], format="%Y-%m-%d %H:%M:%S")
    price_data["time_pd"] = pd.to_datetime(price_data["time"], utc=True)
    price_data = price_data.set_index("time_pd")
    
    return price_data#[date_begin+" 00:00:00+00:00": date_end+" 00:00:00+00:00"]


def get_price_usd_data_bitquery(
    token_address,
    date_begin,
    date_end,
    api_token,
    file_name,
    DOWNLOAD_DATA=True,
    RATE_LIMIT=False,
    exchange_to_query="Uniswap",
):
    """
    Queries the price history of an ERC20 + USD Stablecoins (located at token_address) in exchange_to_query (defaults to all Uniswap versions on mainnet) between begin_date and end_date on Bitquery.
    """

    request = []
    max_rows_bitquery = 10000

    if DOWNLOAD_DATA:
        # Paginate using limit and an offset
        offset = 0
        current_request = run_bitquery_query(
            generate_usd_price_payload(
                token_address, date_begin, date_end, offset, exchange_to_query
            ),
            api_token,
        )
        request.append(current_request)

        # When a request has less than 10,000 rows we are at the last one
        while (
            len(current_request["data"]["ethereum"]["dexTrades"]) == max_rows_bitquery
        ):
            current_request = run_bitquery_query(
                generate_usd_price_payload(
                    token_address, date_begin, date_end, offset, exchange_to_query
                ),
                api_token,
            )
            request.append(current_request)
            offset += max_rows_bitquery
            if RATE_LIMIT:
                time.sleep(5)

        with open("./data/" + file_name + "_1min.pkl", "wb") as output:
            pickle.dump(request, output, pickle.HIGHEST_PROTOCOL)
    else:
        with open("./data/" + file_name + "_1min.pkl", "rb") as input:
            request = pickle.load(input)

    # Prepare data for strategy:
    # Collect json data and add to a pandas Data Frame

    requests_with_data = [len(x["data"]["ethereum"]["dexTrades"]) > 0 for x in request]
    relevant_requests = list(compress(request, requests_with_data))

    price_data = pd.concat(
        [
            pd.DataFrame(
                {
                    "time": [
                        x["timeInterval"]["minute"]
                        for x in request_price["data"]["ethereum"]["dexTrades"]
                    ],
                    "baseCurrency": [
                        x["baseCurrency"]["symbol"]
                        for x in request_price["data"]["ethereum"]["dexTrades"]
                    ],
                    "quoteCurrency": [
                        x["quoteCurrency"]["symbol"]
                        for x in request_price["data"]["ethereum"]["dexTrades"]
                    ],
                    "quoteAmount": [
                        x["quoteAmount"]
                        for x in request_price["data"]["ethereum"]["dexTrades"]
                    ],
                    "baseAmount": [
                        x["baseAmount"]
                        for x in request_price["data"]["ethereum"]["dexTrades"]
                    ],
                    "quotePrice": [
                        x["quotePrice"]
                        for x in request_price["data"]["ethereum"]["dexTrades"]
                    ],
                }
            )
            for request_price in relevant_requests
        ]
    )

    price_data["time"] = pd.to_datetime(price_data["time"], format="%Y-%m-%d %H:%M:%S")
    price_data["time_pd"] = pd.to_datetime(price_data["time"], utc=True)
    price_data = price_data.set_index("time_pd")

    return price_data


def generate_price_payload(
    token_0_address,
    token_1_address,
    date_begin,
    date_end,
    offset,
    exchange_to_query="Uniswap",
):
    payload = (
        """{
                  ethereum(network: ethereum) {
                    dexTrades(
                      options: {asc: "timeInterval.minute", limit: 10000, offset:"""
        + str(offset)
        + '''}
                      date: {between: ["'''
        + date_begin
        + '''","'''
        + date_end
        + '''"]}
                      exchangeName: {is: "'''
        + exchange_to_query
        + '''"}
                      baseCurrency: {is: "'''
        + token_0_address
        + '''"}
                      quoteCurrency: {is: "'''
        + token_1_address
        + """"}

                    ) {
                      timeInterval {
                        minute(count: 1)
                      }
                      baseCurrency {
                        symbol
                        address
                      }
                      baseAmount
                      quoteCurrency {
                        symbol
                        address
                      }
                      tradeAmount(in: USD)
                      quoteAmount
                      quotePrice
                    }
                  }
                }"""
    )

    return payload


def generate_usd_price_payload(
    token_address, date_begin, date_end, offset, exchange_to_query="Uniswap"
):
    payload = (
        """{
                  ethereum(network: ethereum) {
                    dexTrades(
                      options: {asc: "timeInterval.minute", limit: 10000, offset:"""
        + str(offset)
        + '''}
                      date: {between: ["'''
        + date_begin
        + '''","'''
        + date_end
        + '''"]}
                      exchangeName: {is: "'''
        + exchange_to_query
        + '''"}
                      any: [{baseCurrency: {is: "0xa0b86991c6218b36c1d19d4a2e9eb0ce3606eb48"},
                             quoteCurrency:{is: "'''
        + token_address
        + '''"}},
                            {baseCurrency: {is: "0xdac17f958d2ee523a2206206994597c13d831ec7"},
                             quoteCurrency:{is: "'''
        + token_address
        + """"}}]

                    ) {
                      timeInterval {
                        minute(count: 1)
                      }
                      baseCurrency {
                        symbol
                        address
                      }
                      baseAmount
                      quoteCurrency {
                        symbol
                        address
                      }
                      quoteAmount
                      quotePrice
                    }
                  }
                }"""
    )

    return payload


def run_bitquery_query(query, api_token):
    """
    Internal function that runs a GraphQL query on Bitquery.
    """
    url = "https://graphql.bitquery.io/"
    headers = {"X-API-KEY": api_token}
    request = requests.post(url, json={"query": query}, headers=headers)
    if request.status_code == 200:
        return request.json()
    else:
        raise Exception(
            "Query failed and return code is {}.      {}".format(
                request.status_code, query
            )
        )


def get_current_state() -> dict:
    query = '''{
        pool(id: "0x8ad599c3a0ff1de082011efddc58f1908eb6e6d8") {
        tick
    token0
    {
        symbol
    id
    decimals
    }
    token1
    {
        symbol
    id
    decimals
    }
    feeTier
    sqrtPrice
    liquidity
    }
    }'''
    univ3_graph_url = "https://api.thegraph.com/subgraphs/name/uniswap/uniswap-v3"
    params = {"query": query}
    response = requests.post(univ3_graph_url, json=params)
    return response.json()