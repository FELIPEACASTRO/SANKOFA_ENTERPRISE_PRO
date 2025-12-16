"""
CHECKPOINT 3 - Contract Tests: CacheBackend
MLOpsPlatform_QA + BackendEngineer_QA

Contrato ABC CacheBackend:
- get(key: str) -> Optional[Any]
- set(key: str, value: Any, ttl: Optional[int]) -> bool
- delete(key: str) -> bool
- exists(key: str) -> bool
- clear() -> int
- health_check() -> bool

Implementações testadas:
- InMemoryBackend
- (RedisBackend - se disponível)
"""
import pytest
import asyncio
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..', '..'))


class TestCacheBackendContract:
    """Contract tests para qualquer implementação de CacheBackend"""

    @pytest.fixture
    def inmemory_backend(self):
        """Fixture para InMemoryBackend"""
        from backend.infrastructure.cache import InMemoryBackend
        return InMemoryBackend(max_size=100, default_ttl=60)

    @pytest.mark.asyncio
    async def test_CONTRACT_001_get_returns_none_for_missing_key(self, inmemory_backend):
        """CONTRACT-001: get() deve retornar None para chave inexistente"""
        result = await inmemory_backend.get("nonexistent_key")
        assert result is None, "get() deve retornar None para chave inexistente"

    @pytest.mark.asyncio
    async def test_CONTRACT_002_set_returns_bool(self, inmemory_backend):
        """CONTRACT-002: set() deve retornar bool"""
        result = await inmemory_backend.set("test_key", "test_value")
        assert isinstance(result, bool), f"set() deve retornar bool, got {type(result)}"

    @pytest.mark.asyncio
    async def test_CONTRACT_003_get_returns_set_value(self, inmemory_backend):
        """CONTRACT-003: get() deve retornar valor definido por set()"""
        await inmemory_backend.set("my_key", "my_value")
        result = await inmemory_backend.get("my_key")
        assert result == "my_value", f"Expected 'my_value', got {result}"

    @pytest.mark.asyncio
    async def test_CONTRACT_004_delete_returns_bool(self, inmemory_backend):
        """CONTRACT-004: delete() deve retornar bool"""
        await inmemory_backend.set("key_to_delete", "value")
        result = await inmemory_backend.delete("key_to_delete")
        assert isinstance(result, bool), f"delete() deve retornar bool, got {type(result)}"

    @pytest.mark.asyncio
    async def test_CONTRACT_005_delete_removes_key(self, inmemory_backend):
        """CONTRACT-005: delete() deve remover a chave"""
        await inmemory_backend.set("key_to_delete", "value")
        await inmemory_backend.delete("key_to_delete")
        result = await inmemory_backend.get("key_to_delete")
        assert result is None, "Chave deveria ter sido removida"

    @pytest.mark.asyncio
    async def test_CONTRACT_006_exists_returns_bool(self, inmemory_backend):
        """CONTRACT-006: exists() deve retornar bool"""
        result = await inmemory_backend.exists("any_key")
        assert isinstance(result, bool), f"exists() deve retornar bool, got {type(result)}"

    @pytest.mark.asyncio
    async def test_CONTRACT_007_exists_true_for_existing_key(self, inmemory_backend):
        """CONTRACT-007: exists() deve retornar True para chave existente"""
        await inmemory_backend.set("existing_key", "value")
        result = await inmemory_backend.exists("existing_key")
        assert result == True, "exists() deve retornar True para chave existente"

    @pytest.mark.asyncio
    async def test_CONTRACT_008_exists_false_for_missing_key(self, inmemory_backend):
        """CONTRACT-008: exists() deve retornar False para chave inexistente"""
        result = await inmemory_backend.exists("nonexistent_key")
        assert result == False, "exists() deve retornar False para chave inexistente"

    @pytest.mark.asyncio
    async def test_CONTRACT_009_clear_returns_int(self, inmemory_backend):
        """CONTRACT-009: clear() deve retornar int"""
        await inmemory_backend.set("key1", "value1")
        await inmemory_backend.set("key2", "value2")
        result = await inmemory_backend.clear()
        assert isinstance(result, int), f"clear() deve retornar int, got {type(result)}"

    @pytest.mark.asyncio
    async def test_CONTRACT_010_clear_removes_all_keys(self, inmemory_backend):
        """CONTRACT-010: clear() deve remover todas as chaves"""
        await inmemory_backend.set("key1", "value1")
        await inmemory_backend.set("key2", "value2")
        await inmemory_backend.clear()

        exists1 = await inmemory_backend.exists("key1")
        exists2 = await inmemory_backend.exists("key2")

        assert exists1 == False, "key1 deveria ter sido removida"
        assert exists2 == False, "key2 deveria ter sido removida"

    @pytest.mark.asyncio
    async def test_CONTRACT_011_health_check_returns_bool(self, inmemory_backend):
        """CONTRACT-011: health_check() deve retornar bool"""
        result = await inmemory_backend.health_check()
        assert isinstance(result, bool), f"health_check() deve retornar bool, got {type(result)}"

    @pytest.mark.asyncio
    async def test_CONTRACT_012_set_with_ttl(self, inmemory_backend):
        """CONTRACT-012: set() deve aceitar ttl opcional"""
        try:
            result = await inmemory_backend.set("ttl_key", "value", ttl=30)
            assert isinstance(result, bool), "set() com ttl deve retornar bool"
        except TypeError as e:
            pytest.fail(f"set() deveria aceitar parâmetro ttl: {e}")

    @pytest.mark.asyncio
    async def test_CONTRACT_013_set_complex_types(self, inmemory_backend):
        """CONTRACT-013: set() deve aceitar tipos complexos"""
        test_dict = {"nested": {"data": [1, 2, 3]}}
        await inmemory_backend.set("complex_key", test_dict)
        result = await inmemory_backend.get("complex_key")
        assert result == test_dict, "Deve preservar estrutura de dados complexos"


class TestCacheBackendAbstract:
    """Verifica que CacheBackend é realmente abstrato"""

    def test_ABSTRACT_001_cannot_instantiate_directly(self):
        """ABSTRACT-001: Não deve ser possível instanciar CacheBackend diretamente"""
        from backend.infrastructure.cache import CacheBackend

        with pytest.raises(TypeError):
            CacheBackend()

    def test_ABSTRACT_002_has_abstract_methods(self):
        """ABSTRACT-002: Deve ter métodos abstratos definidos"""
        from backend.infrastructure.cache import CacheBackend
        import inspect

        abstract_methods = []
        for name, method in inspect.getmembers(CacheBackend, predicate=inspect.isfunction):
            if getattr(method, '__isabstractmethod__', False):
                abstract_methods.append(name)

        expected_methods = ['get', 'set', 'delete', 'exists', 'clear', 'health_check']
        for method in expected_methods:
            assert method in abstract_methods, f"Método abstrato '{method}' não encontrado"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
