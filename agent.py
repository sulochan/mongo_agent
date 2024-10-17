# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.
# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed in accordance with the terms of the Llama 3 Community License Agreement.
import asyncio

import fire

# from llama_stack_client.agents import *  # noqa: F403
from dotenv import load_dotenv
from .mongo_data import AlertsDataTool
from mongo_agent.common.client_utils import *  # noqa: F403

from .multi_turn import *  # noqa: F403


def main(host: str, port: int, disable_safety: bool = True):
    api_keys = load_api_keys_from_env()
    custom_tools = [AlertsDataTool()]
    agent_config = asyncio.run(
        make_agent_config_with_custom_tools(
            model="Llama3.1-8B-Instruct",
            tool_config=QuickToolConfig(
                custom_tools=custom_tools,
                prompt_format="function_tag",
            ),
            disable_safety=disable_safety,
        )
    )
    asyncio.run(
        execute_turns(
            agent_config=agent_config,
            custom_tools=custom_tools,
            turn_inputs=[
                prompt_to_turn(
                    "How many alerts are there right now with status triggered ?"
                ),
            ],
            host=host,
            port=port,
        )
    )


if __name__ == "__main__":
    fire.Fire(main)
