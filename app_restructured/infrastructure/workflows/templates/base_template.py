from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from enum import Enum


class StepType(Enum):
    """Type of workflow step"""
    TOOL_CALL = "tool_call"           # Direct MCP tool call
    AI_DECISION = "ai_decision"       # AI makes decision about parameters
    AUTONOMOUS = "autonomous"         # Full AI autonomy for this step
    VALIDATION = "validation"         # Validate results before continuing


@dataclass 
class TemplateStep:
    """Defines a single step in a workflow template"""
    name: str                                    # Human readable step name
    step_type: StepType                         # Type of step
    tool_call: Optional[str] = None             # MCP tool to call (if tool_call type)
    parameters: Optional[Dict[str, Any]] = None  # Fixed parameters
    ai_decision_key: Optional[str] = None       # Key for AI decision making
    description: Optional[str] = None           # Description of what this step does
    required: bool = True                       # Whether step is required
    retry_on_failure: bool = True               # Whether to retry failed step
    max_retries: int = 2                        # Maximum retry attempts
    
    def __post_init__(self):
        """Validate step configuration"""
        if self.step_type == StepType.TOOL_CALL and not self.tool_call:
            raise ValueError("TOOL_CALL steps must specify tool_call")
        if self.step_type == StepType.AI_DECISION and not self.ai_decision_key:
            raise ValueError("AI_DECISION steps must specify ai_decision_key")


class BaseTemplate(ABC):
    """Base class for all issue implementation templates"""
    
    def __init__(self):
        self.name = self.__class__.__name__.replace("Template", "").lower()
        self.display_name = self.name.replace("_", " ").title()
    
    @abstractmethod
    def get_discovery_steps(self) -> List[TemplateStep]:
        """Steps to gather context and understand the issue"""
        pass
    
    @abstractmethod  
    def get_planning_steps(self) -> List[TemplateStep]:
        """Steps to create implementation plan"""
        pass
    
    @abstractmethod
    def get_implementation_steps(self) -> List[TemplateStep]:
        """Steps to implement the solution"""
        pass
    
    @abstractmethod
    def get_validation_steps(self) -> List[TemplateStep]:
        """Steps to validate and finalize the implementation"""
        pass
    
    @abstractmethod
    def is_applicable(self, issue_context: Dict[str, Any]) -> float:
        """
        Return confidence score (0.0-1.0) if this template applies to the issue.
        
        Args:
            issue_context: Dictionary containing issue data including:
                - title: Issue title
                - description: Issue description
                - labels: List of labels
                - assignees: List of assignees
                - milestone: Milestone info
                - notes: Issue comments/notes
        
        Returns:
            float: Confidence score from 0.0 to 1.0
        """
        pass
    
    def get_all_steps(self) -> List[TemplateStep]:
        """Get all template steps in execution order"""
        steps = []
        steps.extend(self.get_discovery_steps())
        steps.extend(self.get_planning_steps())
        steps.extend(self.get_implementation_steps())
        steps.extend(self.get_validation_steps())
        return steps
    
    def get_estimated_duration_minutes(self) -> int:
        """Estimate duration in minutes for this template"""
        base_times = {
            StepType.TOOL_CALL: 1,      # 1 minute per tool call
            StepType.AI_DECISION: 2,    # 2 minutes for AI decisions
            StepType.AUTONOMOUS: 10,    # 10 minutes for autonomous work
            StepType.VALIDATION: 3      # 3 minutes for validation
        }
        
        total_minutes = 0
        for step in self.get_all_steps():
            total_minutes += base_times.get(step.step_type, 5)
        
        return total_minutes
    
    def get_complexity_score(self) -> float:
        """Get complexity score (0.0-1.0) based on template steps"""
        complexity_weights = {
            StepType.TOOL_CALL: 0.1,
            StepType.AI_DECISION: 0.2,
            StepType.AUTONOMOUS: 0.4,
            StepType.VALIDATION: 0.1
        }
        
        total_complexity = 0.0
        steps = self.get_all_steps()
        
        for step in steps:
            total_complexity += complexity_weights.get(step.step_type, 0.2)
        
        # Normalize by number of steps
        if steps:
            return min(total_complexity / len(steps), 1.0)
        return 0.0
    
    def get_info(self) -> Dict[str, Any]:
        """Get template information for UI display"""
        steps = self.get_all_steps()
        return {
            "name": self.name,
            "display_name": self.display_name,
            "total_steps": len(steps),
            "estimated_duration_minutes": self.get_estimated_duration_minutes(),
            "complexity_score": self.get_complexity_score(),
            "step_breakdown": {
                "discovery": len(self.get_discovery_steps()),
                "planning": len(self.get_planning_steps()), 
                "implementation": len(self.get_implementation_steps()),
                "validation": len(self.get_validation_steps())
            }
        }


class TemplateRegistry:
    """Registry for managing available templates"""
    
    def __init__(self):
        self._templates: Dict[str, BaseTemplate] = {}
    
    def register(self, template: BaseTemplate):
        """Register a template"""
        self._templates[template.name] = template
    
    def get_template(self, name: str) -> Optional[BaseTemplate]:
        """Get template by name"""
        return self._templates.get(name)
    
    def get_all_templates(self) -> Dict[str, BaseTemplate]:
        """Get all registered templates"""
        return self._templates.copy()
    
    def find_best_template(self, issue_context: Dict[str, Any]) -> Optional[BaseTemplate]:
        """
        Find the best template for an issue based on confidence scores.
        
        Args:
            issue_context: Issue context data
            
        Returns:
            BaseTemplate: Best matching template or None
        """
        best_template = None
        best_score = 0.0
        
        for template in self._templates.values():
            score = template.is_applicable(issue_context)
            if score > best_score:
                best_score = score
                best_template = template
        
        # Only return template if confidence is above threshold
        if best_score >= 0.5:
            return best_template
        return None
    
    def get_template_info(self) -> List[Dict[str, Any]]:
        """Get information about all templates for API responses"""
        return [
            {**template.get_info(), "confidence_threshold": 0.5}
            for template in self._templates.values()
        ]


# Global template registry instance
template_registry = TemplateRegistry()


def get_available_templates() -> List[Dict[str, Any]]:
    """Get list of available templates"""
    return template_registry.get_template_info()


def register_template(template: BaseTemplate):
    """Register a template with the global registry"""
    template_registry.register(template)