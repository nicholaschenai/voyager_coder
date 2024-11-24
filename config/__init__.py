VOYAGER_CONFIG = {
    'memories': [
        {
            'module': 'voyager_coder.cog_arch.memories.voyager_procedural_mem',
            'class': 'VoyagerProceduralMem',
            'name': 'procedural_mem'
        }
    ],
    'reasoning': [
        {
            'module': 'voyager_coder.cog_arch.reasoning.voyager_coding',
            'class': 'VoyagerCodingModule',
            'name': 'coding_module'
        },
        {
            'module': 'voyager_coder.cog_arch.reasoning.voyager_curriculum',
            'class': 'VoyagerCurriculumModule',
            'name': 'curriculum_module'
        },
        {
            'module': 'voyager_coder.cog_arch.reasoning.critic',
            'class': 'CriticModule',
            'name': 'critic_module'
        },
        {
            'module': 'voyager_coder.cog_arch.reasoning.voyager_skill',
            'class': 'VoyagerSkill',
            'name': 'desc_module'
        }
    ],
    'decisions': {
        'module': 'voyager_coder.cog_arch.agents.voyager_agent',
        'class': 'VoyagerAgent'
    }
}
