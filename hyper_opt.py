import math

import click
import numpy as np
from hyperopt import STATUS_OK, Trials, fmin, hp, tpe
from hyperopt.pyll import scope

from tools.trainArg import main as train_mmpre

# same approach can be applied using train_mmseg, train_mmdet


def objective(params):
    global epoch_hyperopt
    opt_type = params.get("opt_type")
    lr = params.get("lr")
    momentum = params.get("momentum")
    weight_decay = params.get("weight_decay")
    # config = 'configs/hexa/scnet_r101_fpn_1x_coco_v2.py' # Get from system Arg
    loss = train_mmpre(
        config_file,
        epoch=epoch_hyperopt,
        opt_type=opt_type,
        lr=lr,
        momentum=momentum,
        weight_decay=weight_decay,
    )
    epoch_hyperopt += 1
    return {"loss": loss, "status": STATUS_OK}


def customStopCondition(x, *kwargs):
    return math.isnan(float(x.best_trial["result"]["loss"])), kwargs


@click.command()
@click.argument("config")
def main(config):
    global epoch_hyperopt
    epoch_hyperopt = 0

    global config_file
    config_file = config

    search_space = hp.choice(
        "otimizer",
        [
            {
                "opt_type": "SGD",
                "lr": hp.loguniform("lr_S", np.log(2 * 1e-4), np.log(2 * 1e-2)),
                "momentum": hp.uniform("momentum_S", 0.3, 0.9),
                "weight_decay": hp.loguniform(
                    "weight_decay_S", np.log(1e-5), np.log(1e-2)
                ),
            },
            {
                "opt_type": "RMSprop",
                "lr": hp.loguniform("lr_A", np.log(5 * 1e-6), np.log(2 * 1e-4)),
                "momentum": hp.uniform("momentum_A", 0, 1e-3),
                "weight_decay": hp.loguniform(
                    "weight_decay_A", np.log(1e-5), np.log(1e-2)
                ),
            },
        ],
    )

    best_result = fmin(
        fn=objective,
        space=search_space,
        algo=tpe.suggest,
        max_evals=100,
        trials=Trials(),
        early_stop_fn=customStopCondition,
    )

    print(best_result)
    return best_result


if __name__ == "__main__":
    main()
