import click


def option(*deco_args, **deco_kwargs):
    if 'help' in deco_kwargs:
        help_doc = deco_kwargs['help']
        if 'default' in deco_kwargs:
            default_val = deco_kwargs['default']
            help_doc = f'{help_doc} (Default: {default_val})'
        deco_kwargs['help'] = help_doc

    return click.option(
        *deco_args,
        **deco_kwargs
    )
