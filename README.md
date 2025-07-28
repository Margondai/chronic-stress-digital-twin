# Engineering Mental Wellness: A Digital Twin for Chronic Stress Modeling and Real-Time Intervention

## Abstract

This project presents a comprehensive digital twin framework for modeling chronic stress progression in adolescents through advanced agent-based simulation. The system integrates trauma history, adverse childhood experiences (ACEs), chronic health conditions, genetic vulnerabilities, emotion regulation, gender identity, and environmental stressors to provide real-time stress assessment and personalized intervention recommendations.

## Overview

Chronic stress significantly impacts mental and physical health among college students and adolescents, yet existing assessment tools rely heavily on self-reporting, which is prone to bias and fails to capture the real-time, fluctuating nature of stress. This research addresses the gap between digital twin theory and practical mental health implementation by comparing three paradigms: Static Baseline Modeling, Adaptive Perfect Implementation, and Realistic Implementation.

### Key Features

- **Three-Tier Simulation Framework**: Static baseline, adaptive learning, and realistic implementation models
- **Agent-Based Modeling**: 100 virtual agents with comprehensive psychological and physiological profiles
- **Real-World Data Integration**: State care indicators, WESAD physiological data, and trauma assessments
- **Intervention Evaluation**: CBT, Mindfulness, and Breathing techniques with effectiveness tracking
- **Clinical Validation**: 70.8% state transition accuracy across 3,000,000 observations

## Research Results

### Therapy Effectiveness Metrics

- **Static Model**: All interventions achieved 0% dropout risk (vs 64% control)
- **Adaptive Model**: 87.5% reduction in control group dropout through personalization
- **Realistic Model**: Robust intervention effectiveness despite real-world constraints
- **Mindfulness**: Most stable intervention across all implementation tiers
- **CBT**: Strong resilience under realistic deployment conditions
- **Breathing Techniques**: Effective but sensitive to personalization algorithms

The system demonstrated that adaptive personalization significantly improves outcomes even without formal therapeutic intervention, with mindfulness showing the greatest stability across implementation conditions.

## System Architecture

The digital twin framework consists of three main components:

1. **Multi-Modal Data Integration**: ABCD study demographics, WESAD physiological data, state care accessibility, and ACE scores
2. **Agent-Based Simulation Engine**: 100 virtual agents across four groups (Control, CBT, Mindfulness, Breathing)
3. **Clinical Validation Framework**: Real-time prediction accuracy and intervention effectiveness tracking

## Installation

### Prerequisites

- Python 3.8 or higher
- Required dependencies listed in `requirements.txt`

### Setup

```bash
git clone https://github.com/yourusername/chronic-stress-digital-twin.git
cd chronic-stress-digital-twin
pip install -r requirements.txt
python affective_state_simulation.py
```

## Usage

### Basic Usage

Run the complete three-tier simulation:

```bash
python affective_state_simulation.py
```

The system will automatically:
- Generate 100 virtual agents with realistic psychological profiles
- Run three parallel simulations (Static, Adaptive, Realistic)
- Execute 30,000 time steps (~2.5 years of daily interactions)
- Generate comprehensive analysis and visualizations
- Export results to CSV files and statistical summaries

### Simulation Parameters

- **Agents**: 100 virtual adolescents (ages 6-18, 50% female)
- **Duration**: 30,000 steps (~2.5 years)
- **Interventions**: CBT (every 7 days), Mindfulness (every 3 days), Breathing (every 2 days)
- **States**: Calm, Mild, Moderate, Severe, Recovered, Deceased
- **Data Points**: 3,000,000 total observations

## Methodology

### Three-Tier Framework

**Tier 1: Static Baseline Model**
- Fixed transition probabilities from literature
- Set intervention intervals without adaptation
- Idealized conditions for baseline comparison

**Tier 2: Adaptive Digital Twin Model**
- 100-step anxiety and 50-step state history tracking
- Resilience factors adjusting from 0.5 to 1.5 based on trends
- Personalized transition probabilities
- Intervention optimization based on effectiveness history

**Tier 3: Realistic Implementation Model**
- Adherence decay (1-3% monthly)
- Technical barriers (3-10% probability)
- Life disruptions and engagement fatigue
- Real-world deployment constraints

### Agent Characteristics

Each virtual agent includes:
- **Demographics**: Age, gender, geographic location
- **Psychological**: Baseline anxiety, emotional regulation, ACE scores
- **Physiological**: HRV patterns, stress sensitivity markers
- **Environmental**: Social support, life stress levels, genetic vulnerability

## File Structure

```
chronic-stress-digital-twin/
├── affective_state_simulation.py  # Main simulation system
├── README.md                      # Project documentation
├── LICENSE                        # MIT license
├── requirements.txt               # Python dependencies
├── setup.py                       # Package installation
├── .gitignore                     # Git ignore rules
└── MODSIM_2025_paper_64.pdf      # Research paper
```

## Results and Analysis

The system generates comprehensive outputs including:

- **Statistical Analysis**: ANOVA, t-tests, effect sizes for intervention comparisons
- **Visualizations**: Anxiety trends, state distributions, sensitivity analyses
- **Clinical Metrics**: Dropout risk, calm state occupancy, intervention effectiveness
- **Validation Reports**: Prediction accuracy, false positive/negative rates

### Key Findings

1. **Adaptive personalization** reduced dropout risk by 87.5% in control groups
2. **Mindfulness interventions** showed greatest stability across all conditions
3. **Real-world constraints** reduced effect sizes but maintained clinical significance
4. **Individual adaptation** proved more impactful than between-group differences

## Clinical Applications

This framework enables:

- **Risk Prediction**: Real-time assessment of stress escalation
- **Intervention Optimization**: Personalized therapy recommendations
- **Clinical Decision Support**: Evidence-based treatment planning
- **Scalable Mental Health**: Cost-effective population-level interventions

## Dataset Sources

- **ABCD Study**: Demographics and developmental data
- **WESAD**: Physiological stress and affect detection
- **NSCH 2022**: State-level care accessibility indicators
- **Literature-Based**: ACE scores, intervention effect sizes

## Testing

The framework includes built-in validation:

```bash
# Statistical validation runs automatically during simulation
python affective_state_simulation.py
```

Validation metrics:
- State transition accuracy: 70.8%
- Anxiety prediction correlation: r = 0.94
- False positive/negative rates: 4.9% each

## Contributing

This is an academic research project. For contributions or collaborations, please contact the authors directly.

## Conference Presentation

This work was presented at MODSIM World 2025 (Paper No. 77). The complete paper and supplementary materials are included in this repository.

## Citation

If you use this software in your research, please cite:

```
Ezcurra, V., Margondai, A., Von Ahlefeldt, C., Willox, S., Hani, S., & Mouloua, M. (2025). 
Engineering Mental Wellness: A Digital Twin for Chronic Stress Modeling and Real-Time Intervention. 
MODSIM World 2025, Orlando, FL.
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Contact

For questions about this research, please contact:

- Valentina Ezcurra: Valentinaezcurrabattro@gmail.com
- Ancuta Margondai: Ancuta.Margondai@ucf.edu
- Dr. Mustapha Mouloua: Mustapha.Mouloua@ucf.edu

University of Central Florida  
Orlando, Florida

## Acknowledgments

This research was conducted at the University of Central Florida Human Factors & Cognitive Psychology Lab. Special thanks to the MODSIM World 2025 conference organization and the research participants who contributed to validation datasets.
