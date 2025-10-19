import torch


def compute_alpha(beta, t):
    beta = torch.cat([torch.zeros(1).to(beta.device), beta], dim=0)
    a = (1 - beta).cumprod(dim=0).index_select(0, t + 1).view(-1, 1, 1, 1)
    return a


def generalized_steps(x, seq, model, b, with_t=False, **kwargs):
    with torch.no_grad():
        n = x.size(0)
        seq_next = [-1] + list(seq[:-1])
        x0_preds = []
        xs = [x]
        ts = []
        
        # Create a deterministic random generator if seed is provided
        generator = kwargs.get("generator", None)
        use_deterministic_noise = kwargs.get("deterministic_noise", False)
        base_seed = kwargs.get("base_seed", None)
        
        for step_idx, (i, j) in enumerate(zip(reversed(seq), reversed(seq_next))):
            t = (torch.ones(n) * i).to(x.device)
            next_t = (torch.ones(n) * j).to(x.device)
            at = compute_alpha(b, t.long())
            at_next = compute_alpha(b, next_t.long())
            xt = xs[-1].to('cuda')
            et = model(xt, t)
            x0_t = (xt - et * (1 - at).sqrt()) / at.sqrt()
            x0_preds.append(x0_t.to('cpu'))
            c1 = (
                kwargs.get("eta", 0) * ((1 - at / at_next) * (1 - at_next) / (1 - at)).sqrt()
            )
            c2 = ((1 - at_next) - c1 ** 2).sqrt()
            
            # Generate noise with deterministic seeding if enabled
            if use_deterministic_noise and base_seed is not None:
                # Use per-step seeding to ensure reproducibility across different implementations
                step_generator = torch.Generator(device=x.device)
                step_generator.manual_seed(base_seed + step_idx)
                noise = torch.randn(x.shape, generator=step_generator, device=x.device, dtype=x.dtype)
            elif generator is not None:
                noise = torch.randn(x.shape, generator=generator, device=x.device, dtype=x.dtype)
            else:
                noise = torch.randn_like(x)
            
            xt_next = at_next.sqrt() * x0_t + c1 * noise + c2 * et
            xs.append(xt_next.to('cpu'))
            ts.append(t.to('cpu'))

    if with_t:
        return xs, ts, x0_preds
    else:
        return xs, x0_preds


def ddpm_steps(x, seq, model, b, **kwargs):
    with torch.no_grad():
        n = x.size(0)
        seq_next = [-1] + list(seq[:-1])
        xs = [x]
        x0_preds = []
        betas = b
        
        # Create a deterministic random generator if seed is provided
        generator = kwargs.get("generator", None)
        use_deterministic_noise = kwargs.get("deterministic_noise", False)
        base_seed = kwargs.get("base_seed", None)
        
        for step_idx, (i, j) in enumerate(zip(reversed(seq), reversed(seq_next))):
            t = (torch.ones(n) * i).to(x.device)
            next_t = (torch.ones(n) * j).to(x.device)
            at = compute_alpha(betas, t.long())
            atm1 = compute_alpha(betas, next_t.long())
            beta_t = 1 - at / atm1
            x = xs[-1].to('cuda')

            output = model(x, t.float())
            e = output

            x0_from_e = (1.0 / at).sqrt() * x - (1.0 / at - 1).sqrt() * e
            x0_from_e = torch.clamp(x0_from_e, -1, 1)
            x0_preds.append(x0_from_e.to('cpu'))
            mean_eps = (
                (atm1.sqrt() * beta_t) * x0_from_e + ((1 - beta_t).sqrt() * (1 - atm1)) * x
            ) / (1.0 - at)

            mean = mean_eps
            
            # Generate noise with deterministic seeding if enabled
            if use_deterministic_noise and base_seed is not None:
                step_generator = torch.Generator(device=x.device)
                step_generator.manual_seed(base_seed + step_idx)
                noise = torch.randn(x.shape, generator=step_generator, device=x.device, dtype=x.dtype)
            elif generator is not None:
                noise = torch.randn(x.shape, generator=generator, device=x.device, dtype=x.dtype)
            else:
                noise = torch.randn_like(x)
            
            mask = 1 - (t == 0).float()
            mask = mask.view(-1, 1, 1, 1)
            logvar = beta_t.log()
            sample = mean + mask * torch.exp(0.5 * logvar) * noise
            xs.append(sample.to('cpu'))
    return xs, x0_preds
