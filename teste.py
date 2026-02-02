import asyncio
import json
import logging
from agente2.tools import MCPClient
from dotenv import load_dotenv, find_dotenv

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)

async def test_get_visible_objects():
    load_dotenv(find_dotenv())
    
    mcp = MCPClient("http://localhost:8000/mcp")
    
    try:
        print("Conectando ao MCP...")
        await mcp.connect()
        
        print("Carregando tools...")
        tools = await mcp.load_tools()
        tool_registry = {tool.name: tool.coroutine for tool in tools}
        
        if "rotate_right" not in tool_registry:
            print("ERRO: get_visible_objects não encontrada!")
            return
        
        print("\nChamando pickup_object...")
        object_id = "Ladle|-00.79|+01.09|+02.14"
        result = await tool_registry["get_visible_objects"]()
        
        print("\n=== RESULTADO ===")
        print(f"Tipo: {type(result)}")
        print(f"Conteúdo: {result}")

        # Tenta parsear se for string JSON
        if isinstance(result, list) and len(result) > 0:
            first_item = result[0]
            if isinstance(first_item, str):
                try:
                    parsed = json.loads(first_item)
                    print("\n=== PARSEADO (JSON) ===")
                    print(json.dumps(parsed, indent=2, ensure_ascii=False))
                except:
                    print("\nNão é JSON válido")
        
    finally:
        await mcp.close()
        print("\nConexão fechada")

if __name__ == "__main__":
    asyncio.run(test_get_visible_objects())