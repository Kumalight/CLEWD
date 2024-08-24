from .visda17 import form_visda17
from .visda18 import form_visda18
from .domainnet import form_domainnet
from .office import form_office
from .nwpu45 import form_nwpu45
from .smokersnos10 import form_smokenos10
from .smokersnos30_10 import form_smokenos30_10
# from .smokerscross import form_smokerscross
from .aid256 import form_aid

# Known : unknown ratio
# ano_type = 1 -> 1:10 (competition)
# ano_type = 2 -> 1:1
# ano_type = 3 -> 10:1

def form_visda_datasets(config, ignore_anomaly=False):
    if config.dataset == 'VISDA17':
        return form_visda17(config)
    elif config.dataset == 'VISDA18':
        return form_visda18(config, ignore_anomaly, config.ano_type)
    elif config.dataset == 'DomainNet':
        return form_domainnet(config)
    elif config.dataset == 'office':
        return form_office(config)
    elif config.dataset == '20' or '10' :
        return form_nwpu45(config)
    elif config.dataset == 'smokersbynos10' or 'smokersbynos15'or 'smokersbynos05'or 'smokersbynos20' or 'smokerscls20' or 'smokerscls10' :
        return form_smokenos10(config)
    elif config.dataset == 'smokersbynos30_10':
        return form_smokenos30_10(config)
    elif config.dataset == 'aid20' or 'aid50':
        return form_aid(config)


    else:
        raise ValueError('Please specify a valid dataset | VISDA17 / VISDA18')