import pytest
import torch
import torch.distributed as dist
from transformers.models.bert.modeling_bert import BertModel

import colossalai
from colossalai.cluster import ProcessGroupMesh
from colossalai.pipeline.stage_manager import PipelineStageManager
from colossalai.shardformer.policies.bert import BertModelPolicy, bert_model_forward
from colossalai.shardformer.shard import ShardConfig
from colossalai.testing import rerun_if_address_is_in_use, spawn


def check_bert_model_forward():
    model = BertModel.from_pretrained('bert-base-uncased')
    DP_DIM, PP_DIM = 0, 1
    DP_SIZE, PP_SIZE = 2, 2
    RANK_TO_COORDINATE = {
        0: (0, 0),
        1: (0, 1),
        2: (1, 0),
        3: (1, 1),
    }
    PP_RANKS_IN_GROUP = {
        0: [0, 1],
        1: [0, 1],
        2: [2, 3],
        3: [2, 3],
    }
    pg_mesh = ProcessGroupMesh(DP_SIZE, PP_SIZE)

    # print(pg_mesh)

    stage_manager = PipelineStageManager(pg_mesh, PP_DIM)
    rank = dist.get_rank()
    # print(rank)

    x = torch.randint(0, 1000, (2, 3))
    hidden_states = torch.randint(0, 1000, (2, 3, 768)).to(torch.float32)
    if stage_manager.stage == 0:
        attention_mask = torch.ones_like(x)
        output = bert_model_forward(self=model, input_ids=x, attention_mask=attention_mask, stage_manager=stage_manager)
        print(output['hidden_states'].shape)
        assert output['hidden_states'].shape == (2, 3, 768)
    else:
        attention_mask = torch.ones((2, 3))
        output = bert_model_forward(self=model,
                                    hidden_states=hidden_states,
                                    attention_mask=attention_mask,
                                    stage_manager=stage_manager)
        print(output[0].shape)
        assert output[0].shape == (2, 3, 768)

    # assert output[1].shape == (2, 768)


def check_bert_model_policy():
    model = BertModel.from_pretrained('bert-base-uncased')
    DP_DIM, PP_DIM = 0, 1
    DP_SIZE, PP_SIZE = 2, 2
    RANK_TO_COORDINATE = {
        0: (0, 0),
        1: (0, 1),
        2: (1, 0),
        3: (1, 1),
    }
    PP_RANKS_IN_GROUP = {
        0: [0, 1],
        1: [0, 1],
        2: [2, 3],
        3: [2, 3],
    }
    pg_mesh = ProcessGroupMesh(DP_SIZE, PP_SIZE)
    # print(pg_mesh)

    stage_manager = PipelineStageManager(pg_mesh, PP_DIM)
    rank = dist.get_rank()

    model_policy = BertModelPolicy()
    model_policy.set_model(model)
    model_config = ShardConfig(pipeline_stage_manager=stage_manager, enable_tensor_parallelism=False)
    model_policy.set_shard_config(model_config)

    layers = model_policy.get_held_layers()

    assert layers is not None


def run_dist_model(rank, world_size, port):
    colossalai.launch(config={}, rank=rank, world_size=world_size, port=port, host='localhost')
    check_bert_model_forward()


def run_dist_policy(rank, world_size, port):
    colossalai.launch(config={}, rank=rank, world_size=world_size, port=port, host='localhost')
    check_bert_model_policy()


@pytest.mark.dist
@rerun_if_address_is_in_use()
def test_bert_model_forward():
    spawn(run_dist_model, 4)


@pytest.mark.dist
@rerun_if_address_is_in_use()
def test_bert_model_policy():
    spawn(run_dist_policy, 4)


if __name__ == "__main__":
    """test the bert model forward and bert model policy"""
    #test_bert_model_forward()
    test_bert_model_policy()
    # this test need config to run