import argparse
import os
import torch
from collections import OrderedDict

def get_model_config_from_args(args):
    return {
        'item_num': args.item_num,
        'maxlen': args.max_length,
        'hidden_units': args.hidden_units,
        'num_blocks': args.num_blocks,
        'num_heads': args.num_heads,
        'dropout_rate': args.dropout_rate,
    }

def stack_sasrec_weights(shallow_state_dict, shallow_blocks, deep_blocks):
    if deep_blocks <= shallow_blocks:
        raise ValueError(f"Deep blocks ({deep_blocks}) must be greater than shallow blocks ({shallow_blocks})")

    deep_state_dict = OrderedDict()

    for key, param in shallow_state_dict.items():
        if 'attention_layers.' in key or 'attention_layernorms.' in key or \
           'forward_layers.' in key or 'forward_layernorms.' in key:
            parts = key.split('.')
            block_idx_str = parts[1]
            block_idx = int(block_idx_str)

            deep_state_dict[key] = param.clone()

            new_block_idx = block_idx + shallow_blocks
            if new_block_idx < deep_blocks:
                new_parts = parts[:1] + [str(new_block_idx)] + parts[2:]
                new_key = '.'.join(new_parts)
                deep_state_dict[new_key] = param.clone()
        else:
            deep_state_dict[key] = param.clone()

    return deep_state_dict


def main():
    parser = argparse.ArgumentParser(description="Stack a trained SASRec model to be deeper.")

    parser.add_argument('--input_model_path', type=str, required=True,
                        help="Path to the shallow model's checkpoint file (.pt).")
    parser.add_argument('--shallow_blocks', type=int, required=True,
                        help="The number of blocks in the input shallow model.")
    parser.add_argument('--deep_blocks', type=int, required=True,
                        help="The number of blocks for the new output deep model.")

    parser.add_argument('--output_dir', type=str, default='checkpoints384',
                        help="Directory to save the new stacked model.")
    parser.add_argument('--output_filename', type=str, default=None,
                        help="Filename for the new stacked model. Defaults to 'stacked_b<deep_blocks>.pt'.")

    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print(f"Loading shallow model from: {args.input_model_path}")
    shallow_state_dict = torch.load(args.input_model_path, map_location='cpu')

    print(f"Stacking model from {args.shallow_blocks} blocks to {args.deep_blocks} blocks...")
    deep_state_dict = stack_sasrec_weights(
        shallow_state_dict,
        args.shallow_blocks,
        args.deep_blocks
    )

    if args.output_filename is None:
        output_filename = f'stacked_b{args.deep_blocks}.pt'
    else:
        output_filename = args.output_filename

    output_path = os.path.join(args.output_dir, output_filename)

    print(f"Saving new deep model state dict to: {output_path}")
    torch.save(deep_state_dict, output_path)

    print("Verifying stacked model keys...")
    try:
        from sasrec.model import SASRec
        verification_model = SASRec(item_num=1000, num_blocks=args.deep_blocks)
        missing_keys, unexpected_keys = verification_model.load_state_dict(deep_state_dict, strict=False)

        is_ok = True
        core_missing_keys = [k for k in missing_keys if 'item_emb' not in k and 'pos_emb' not in k]
        if core_missing_keys:
             print(f"ERROR! Missing keys found: {core_missing_keys}")
             is_ok = False
        if unexpected_keys:
            print(f"ERROR! Unexpected keys found: {unexpected_keys}")
            is_ok = False

        if is_ok:
            print(f"SUCCESS! State dict seems compatible with a SASRec model of {args.deep_blocks} blocks.")

    except Exception as e:
        print(f"WARNING! Could not complete full verification due to missing SASRec model definition or other error: {e}")


if __name__ == '__main__':
    main()
