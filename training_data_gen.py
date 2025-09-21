import numpy as np
import math
import json

# Constants
PI = 3.141592
I_STEPS = 16
J_STEPS = 8

def rsi(r0, rd, sr):
    """
    Ray-sphere intersection that assumes the sphere is centered at the origin.
    Returns intersection distances. No intersection when result[0] > result[1].
    
    Args:
        r0: Ray origin (3D vector)
        rd: Ray direction (3D vector) 
        sr: Sphere radius (float)
    
    Returns:
        numpy array with two intersection distances
    """
    a = np.dot(rd, rd)
    b = 2.0 * np.dot(rd, r0)
    c = np.dot(r0, r0) - (sr * sr)
    d = (b * b) - 4.0 * a * c
    
    if d < 0.0:
        return np.array([1e5, -1e5])
    
    sqrt_d = math.sqrt(d)
    return np.array([
        (-b - sqrt_d) / (2.0 * a),
        (-b + sqrt_d) / (2.0 * a)
    ])

def atmosphere(r, r0=np.array([0.0, 6372e3, 0.0]), pSun=np.array([1.0, 0.5, -1]), iSun=22.0, rPlanet=6371e3, rAtmos=6471e3, kRlh=np.array([5.5e-6, 13.0e-6, 22.4e-6]), kMie=21e-6, shRlh=8e3, shMie=1.2e3, g=0.758):
    """
    Calculate atmospheric scattering color.
    
    Args:
        r: View direction (3D vector)
        r0: Ray origin (3D vector)
        pSun: Sun direction (3D vector)
        iSun: Sun intensity (float)
        rPlanet: Planet radius (float)
        rAtmos: Atmosphere radius (float)
        kRlh: Rayleigh scattering coefficient (3D vector)
        kMie: Mie scattering coefficient (float)
        shRlh: Rayleigh scale height (float)
        shMie: Mie scale height (float)
        g: Mie scattering asymmetry factor (float)
    
    Returns:
        Final atmospheric color (3D vector)
    """
    # Normalize the sun and view directions
    pSun = pSun / np.linalg.norm(pSun)
    r = r / np.linalg.norm(r)
    
    # Calculate the step size of the primary ray
    p = rsi(r0, r, rAtmos)
    if p[0] > p[1]:
        return np.array([0.0, 0.0, 0.0])
    
    p[1] = min(p[1], rsi(r0, r, rPlanet)[0])
    iStepSize = (p[1] - p[0]) / float(I_STEPS)
    
    # Initialize the primary ray time
    iTime = 0.0
    
    # Initialize accumulators for Rayleigh and Mie scattering
    totalRlh = np.array([0.0, 0.0, 0.0])
    totalMie = np.array([0.0, 0.0, 0.0])
    
    # Initialize optical depth accumulators for the primary ray
    iOdRlh = 0.0
    iOdMie = 0.0
    
    # Calculate the Rayleigh and Mie phases
    mu = np.dot(r, pSun)
    mumu = mu * mu
    gg = g * g
    pRlh = 3.0 / (16.0 * PI) * (1.0 + mumu)
    pMie = 3.0 / (8.0 * PI) * ((1.0 - gg) * (mumu + 1.0)) / (pow(1.0 + gg - 2.0 * mu * g, 1.5) * (2.0 + gg))
    
    # Sample the primary ray
    for i in range(I_STEPS):
        # Calculate the primary ray sample position
        iPos = r0 + r * (iTime + iStepSize * 0.5)
        
        # Calculate the height of the sample
        iHeight = np.linalg.norm(iPos) - rPlanet
        
        # Calculate the optical depth of the Rayleigh and Mie scattering for this step
        odStepRlh = math.exp(-iHeight / shRlh) * iStepSize
        odStepMie = math.exp(-iHeight / shMie) * iStepSize
        
        # Accumulate optical depth
        iOdRlh += odStepRlh
        iOdMie += odStepMie
        
        # Calculate the step size of the secondary ray
        jStepSize = rsi(iPos, pSun, rAtmos)[1] / float(J_STEPS)
        
        # Initialize the secondary ray time
        jTime = 0.0
        
        # Initialize optical depth accumulators for the secondary ray
        jOdRlh = 0.0
        jOdMie = 0.0
        
        # Sample the secondary ray
        for j in range(J_STEPS):
            # Calculate the secondary ray sample position
            jPos = iPos + pSun * (jTime + jStepSize * 0.5)
            
            # Calculate the height of the sample
            jHeight = np.linalg.norm(jPos) - rPlanet
            
            # Accumulate the optical depth
            jOdRlh += math.exp(-jHeight / shRlh) * jStepSize
            jOdMie += math.exp(-jHeight / shMie) * jStepSize
            
            # Increment the secondary ray time
            jTime += jStepSize
        
        # Calculate attenuation
        attn = np.exp(-(kMie * (iOdMie + jOdMie) + kRlh * (iOdRlh + jOdRlh)))
        
        # Accumulate scattering
        totalRlh += odStepRlh * attn
        totalMie += odStepMie * attn
        
        # Increment the primary ray time
        iTime += iStepSize
    
    # Calculate and return the final color
    return iSun * (pRlh * kRlh * totalRlh + pMie * kMie * totalMie)

def generate_uniform_sphere_points(n):
    """
    Generate n uniformly distributed normalized vectors on the unit sphere.
    
    Args:
        n: Number of points to generate
    
    Returns:
        numpy array of shape (n, 3) with normalized 3D vectors
    """
    # Use the Marsaglia method for uniform distribution on sphere
    points = np.random.normal(0, 1, (n, 3))
    norms = np.linalg.norm(points, axis=1, keepdims=True)
    return points / norms

def main():
    """
    Generate 10,000 random view directions on unit sphere and compute atmospheric scattering.
    Save input and output data to files.
    """
    print("Generating 10,000 atmospheric scattering samples...")
    
    # Earth-like atmosphere parameters
    r0 = np.array([0.0, 6372e3, 0.0])  # Observer position (on surface)
    pSun = np.array([1.0, 0.5, -1])  # Sun direction
    iSun = 22.0  # Sun intensity
    rPlanet = 6371e3  # Earth radius in meters
    rAtmos = 6471e3  # Atmosphere top in meters
    kRlh = np.array([5.5e-6, 13.0e-6, 22.4e-6])  # Rayleigh coefficients
    kMie = 21e-6  # Mie coefficient
    shRlh = 8e3  # Rayleigh scale height
    shMie = 1.2e3  # Mie scale height
    g = 0.758  # Mie asymmetry factor
    
    # Generate 10,000 random view directions on unit sphere
    n_samples = 10000
    view_directions = generate_uniform_sphere_points(n_samples)
    output_colors = np.zeros((n_samples, 3))

    print("Computing atmospheric scattering...")
    for i in range(n_samples):
        if i % 1000 == 0:
            print(f"Progress: {i}/{n_samples}")
        r = view_directions[i]
        color = atmosphere(r, r0, pSun, iSun, rPlanet, rAtmos, kRlh, kMie, shRlh, shMie, g)
        output_colors[i] = color

    # Save as JSON
    print("Saving data to JSON...")
    data = []
    for i in range(n_samples):
        entry = {
            "input": view_directions[i].tolist(),   # [x, y, z]
            "output": output_colors[i].tolist()     # [R, G, B]
        }
        data.append(entry)

    with open("atmosphere_dataset.json", "w") as f:
        json.dump(data, f, indent=2)

    print(f"Saved {n_samples} samples to atmosphere_dataset.json")
    print("Sample entry:", data[0])

if __name__ == "__main__":
    main()
