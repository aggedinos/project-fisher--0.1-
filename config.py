"""Configuration settings for the browser automation agent."""

from dataclasses import dataclass, field

from paths import writable_subdir


@dataclass
class Config:
    """Central configuration for the agent, browser, and LLM providers."""

    
    thinking_level: str = "fast"
    provider: str = "gemini"
    temperature: float = 0.1

    
    @property
    def gemini_model(self) -> str:
        mapping = {
            "fast": "gemini-2.5-flash",
            "thinking": "gemini-2.5-pro",
            "pro": "gemini-3.1-pro"
        }
        return mapping.get(self.thinking_level.lower(), "gemini-2.5-flash")

    
    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "llama3.2-vision"

    
    nvidia_base_url: str = "https://integrate.api.nvidia.com/v1"
    nvidia_model: str = "meta/llama-3.2-90b-vision-instruct"

    
    send_images: bool = True

    
    
    
    click_mode: str = "text"

    
    max_steps: int = 30
    max_json_retries: int = 3
    headless: bool = False
    system_instructions: str = ""
    max_repeated_action: int = 2  
    max_page_text_chars: int = 9000  
    cycle_max_len: int = 8  

    
    max_history_screenshots: int = 3  
    enable_memory_summary: bool = True  

    
    max_step_failures: int = 3  

    
    confidence_pause_below: int = 50   
    confidence_autoskip_below: int = 30  
    supervised: bool = False  

    
    replan_after_stuck: int = 5  

    
    captcha_max_attempts: int = 2  
    captcha_fallback_url: str = "https://duckduckgo.com"

    
    
    
    debug_dir: str = field(default_factory=lambda: writable_subdir("debug"))
    sessions_dir: str = field(default_factory=lambda: writable_subdir("sessions"))

    
    viewport_width: int = 1280
    viewport_height: int = 720

    
    navigate_wait_ms: int = 2500
    click_wait_ms: int = 1200
    type_wait_ms: int = 400
    key_wait_ms: int = 600
    scroll_wait_ms: int = 500
    tab_wait_ms: int = 800

    
    min_action_delay_ms: int = 800
    max_action_delay_ms: int = 2500

    
    use_real_chrome: bool = False
    
    chrome_profile: str = "Default"

    
    default_search_engine: str = "duckduckgo"


config = Config()
