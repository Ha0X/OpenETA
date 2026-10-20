"""Run a registered harness configuration in a private Codex episode."""
from pathlib import Path
import sys

if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts import codex_plugin_smoke as legacy
from tools.harness_config import ConfigError, ConfigRegistry


def parser():
    p = legacy.parser()
    p.description = __doc__
    selection = p.add_mutually_exclusive_group(required=True)
    selection.add_argument('--config-id')
    selection.add_argument('--selection', type=Path, help='Preparation MCP selection artifact')
    selection.add_argument('--prepare-with-model', action='store_true', help='Select configuration in an isolated model preparation session')
    p.add_argument('--preparation-timeout', type=float, default=180)
    p.add_argument('--config-sha256')
    p.add_argument('--episode-action-budget', choices=('unlimited', 'profile'), default='unlimited',
                   help='RoboDojo: unlimited episode actions by default; profile reproduces the selected historical quota')
    p.set_defaults(tool_profile=None, controller=None, timeout=None, max_requests=200)
    return p


def configure(args):
    if args.selection:
        from tools.harness_setup import read_selection
        config = read_selection(args.selection, env_id=args.env_id, task=args.task, seed=args.seed)
    else:
        config = ConfigRegistry().resolve(args.config_id)
    config.bind_task(env_id=args.env_id, task=args.task, seed=args.seed)
    config.require_executable()
    if args.config_sha256 is not None and args.config_sha256 != config.sha256:
        raise ConfigError('config_hash_mismatch', 'Preset changed after selection')
    backend = config.value['components']['benchmark']['backend']
    if args.timeout is None:
        args.timeout = 0.0 if backend == 'robodojo' else 1200.0
    if (backend == 'robodojo' and getattr(args, 'episode_action_budget', 'profile') == 'unlimited'
            and config.value['components']['benchmark'].get('episode_budget_override') != 'robodojo_unlimited_actions_v105'):
        config = ConfigRegistry().resolve(config.id + '__unlimited_v105')
        config.require_executable()
    parts = config.value['components']
    for key, wanted in [('tool_profile', parts['interface']['tool_profile']),
                         ('controller', parts['execution']['controller_option'])]:
        if getattr(args, key) not in (None, wanted):
            raise ConfigError('config_argument_conflict', f'{key} conflicts with {config.id}')
        setattr(args, key, wanted)
    args.harness_config, args.harness_config_sha256 = config.id, config.sha256
    return args


def main():
    p = parser()
    args = p.parse_args()
    if args.output.exists():
        p.error('--output must be a new directory for isolation')
    if args.prepare_with_model:
        from argparse import Namespace
        from scripts.codex_harness_prepare import run as prepare_run
        preparation = args.output.with_name(args.output.name + '-preparation')
        result = prepare_run(Namespace(env_id=args.env_id,task=args.task,seed=args.seed,output=preparation,
            model=args.model,effort=args.effort,model_catalog_json=args.model_catalog_json,
            timeout=args.preparation_timeout,codex=args.codex))
        if result:
            p.error(f'Preparation did not produce a verified binding; see {preparation}/summary.json')
        args.selection = preparation/'selection.json'
    try:
        configure(args)
    except ConfigError as exc:
        p.error(f'{exc.code}: {exc}')
    raise SystemExit(legacy.run(args))


if __name__ == '__main__':
    main()
