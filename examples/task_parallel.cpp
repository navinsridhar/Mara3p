/**
 ==============================================================================
 Copyright 2019, Jonathan Zrake

 Permission is hereby granted, free of charge, to any person obtaining a copy of
 this software and associated documentation files (the "Software"), to deal in
 the Software without restriction, including without limitation the rights to
 use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies
 of the Software, and to permit persons to whom the Software is furnished to do
 so, subject to the following conditions:

 The above copyright notice and this permission notice shall be included in all
 copies or substantial portions of the Software.

 THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 SOFTWARE.

 ==============================================================================
*/




#include <set>
#include "parallel_mpi.hpp"
#include "parallel_message_queue.hpp"
#include "parallel_thread_pool.hpp"
#include "parallel_dependency_graph.hpp"
#include "app_binary_serialize.hpp"




//=============================================================================
template<typename T>
struct serial::container_shape_setter_t<std::vector<T>>
{
    template<typename Serializer>
    auto operator()(Serializer& s, std::vector<T>& value)
    {
        value.resize(s.template vend<std::size_t>());
    }
};

template<typename T>
struct serial::container_shape_descriptor_t<std::vector<T>>
{
    template<typename Serializer>
    auto operator()(Serializer& s, const std::vector<T>& value)
    {
        s(value.size());
    }
};

template<typename T>
struct serial::type_descriptor_t<std::vector<T>>
{
    template<typename Serializer>
    void operator()(Serializer& s, std::vector<T>& value) const
    {
        s(value.data(), value.size());
    }
};

template<>
struct serial::conversion_to_serializable_t<std::string>
{
    using type = std::vector<char>;
    auto operator()(std::string value) const { return std::vector<char>(value.begin(), value.end()); }
};

template<>
struct serial::conversion_from_serializable_t<std::string>
{
    using type = std::vector<char>;
    auto operator()(std::vector<char> value) const { return std::string(value.begin(), value.end()); }
};

template<typename T, typename U>
struct serial::type_descriptor_t<std::pair<T, U>>
{
    template<typename Serializer>
    void operator()(Serializer& s, std::pair<T, U>& value) const
    {
        s(value.first);
        s(value.second);
    }
};




//=============================================================================
template<typename KeyType>
std::map<KeyType, int> partition(const std::vector<KeyType>& keys, unsigned num_groups)
{
    auto result = std::map<KeyType, int>();

    for (std::size_t i = 0; i < num_groups; ++i)
    {
        std::size_t start = (i + 0) * keys.size() / num_groups;
        std::size_t final = (i + 1) * keys.size() / num_groups;

        for (std::size_t j = start; j < final; ++j)
        {
            result[keys[j]] = i;
        }
    }
    return result;
}




//=============================================================================
template<typename KeyType>
std::set<int> recipients(const std::set<KeyType>& keys, const std::map<KeyType, int>& assigned_process)
{
    auto recipients = std::set<int>();

    for (auto key : keys)
        if (auto recipient = assigned_process.at(key); recipient != mpi::comm_world().rank())
            recipients.insert(recipient);

    return recipients;
}




//=========================================================================
template<typename KeyType, typename ValueType, typename ResponsibleForType>
void print_graph_status(const mara::DependencyGraph<KeyType, ValueType>& graph, ResponsibleForType is_responsible_for)
{
    using std::left;
    using std::setw;
    using std::setfill;

    auto& os = std::cout;

    mpi::comm_world().invoke([&] ()
    {
        auto header = "Process "
        + std::to_string(mpi::comm_world().rank())
        + " ("
        + std::to_string(graph.count_unevaluated(is_responsible_for))
        + " unevaluated)";

        os << std::string(52, '=') << "\n";
        os << header << ":\n\n";

        for (auto key : graph.keys())
        {
            os << '\t' << left << setw(24) << setfill('.') << key << ' ';
            os << "status: " << graph.status(key) << " eligible: " << graph.is_eligible(key, is_responsible_for) << (is_responsible_for(key) ? " x" : "");
            os << '\n';
        }
        os << '\n';
    });
}




//=============================================================================
auto build_graph()
{
    auto graph = mara::DependencyGraph<std::string, int>();
    auto mult = [] (auto x) { return x[0] * x[1]; };

    graph.insert_rule("a", [] (auto) { return 1; });
    graph.insert_rule("b", [] (auto) { return 2; });
    graph.insert_rule("c", [] (auto) { return 3; });
    graph.insert_rule("d", [] (auto) { return 4; });
    graph.insert_rule("e", [] (auto) { return 5; });
    graph.insert_rule("f", [] (auto) { return 6; });
    graph.insert_rule("g", [] (auto) { return 7; });
    graph.insert_rule("h", [] (auto) { return 8; });

    graph.insert_rule("ab", mult, "a", "b");
    graph.insert_rule("cd", mult, "c", "d");
    graph.insert_rule("ef", mult, "e", "f");
    graph.insert_rule("gh", mult, "g", "h");

    graph.insert_rule("ae", mult, "a", "e");
    graph.insert_rule("bf", mult, "b", "f");
    graph.insert_rule("cg", mult, "c", "g");
    graph.insert_rule("dh", mult, "d", "h");

    graph.insert_rule("abcd", mult, "ab", "cd");
    graph.insert_rule("efgh", mult, "ef", "gh");

    graph.insert_rule("aebf", mult, "ae", "bf");
    graph.insert_rule("cgdh", mult, "cg", "dh");

    graph.insert_rule("abcdefgh", mult, "abcd", "efgh");
    graph.insert_rule("aebfcgdh", mult, "aebf", "cgdh");

    return graph;
}




// =============================================================================
int main()
{
    MPI_Init_thread(0, nullptr, MPI_THREAD_SERIALIZED, nullptr);


    auto message_queue    = mara::MessageQueue();
    auto scheduler        = mara::ThreadPool();
    auto graph            = build_graph();
    auto assigned         = partition(graph.keys(), mpi::comm_world().size());

    auto is_responsible_for = [&assigned] (std::string key)
    {
        return assigned.at(key) == mpi::comm_world().rank();
    };


    print_graph_status(graph, is_responsible_for);


    while (graph.count_unevaluated(is_responsible_for))
    {
        for (const auto& key : graph.eligible_rules(is_responsible_for))
        {
            graph.evaluate_rule(key, scheduler);                
        }

        for (const auto& [k, p] : graph.poll())
        {
            message_queue.push(serial::dumps(std::pair(k, p)), recipients(graph.downstream_keys(k), assigned));
        }

        for (const auto& received : message_queue.poll())
        {
            graph.insert_product(serial::loads<std::pair<std::string, int>>(received));
        }
    }


    print_graph_status(graph, is_responsible_for);
    mpi::comm_world().invoke([&] ()
    {
        if (is_responsible_for("abcdefgh"))
        {
            std::cout << "The result of abcdefgh is " << graph.product_at("abcdefgh") << std::endl;
        }

        if (is_responsible_for("aebfcgdh"))
        {
            std::cout << "The result of aebfcgdh is " << graph.product_at("aebfcgdh") << std::endl;
        }
    });


    MPI_Finalize();
    return 0;
}